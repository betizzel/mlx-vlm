"""Model-agnostic coordination for Automatic Prefix Caching.

The storage manager owns hashes, blocks, eviction and persistence.  This
coordinator owns the model cache plan and is the only layer generation code
needs to call.  It follows vLLM's split between cache specs/groups and a
coordinator, adapted to MLX's contiguous runtime cache objects.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, List, Optional, Sequence, Tuple

import mlx.core as mx

from .apc_adapters import (
    PrefixCachePlan,
    build_prefix_cache_plan,
    build_prefix_cache_plan_from_caches,
    cache_memory_components,
    clone_cache_entry,
)
from .models.cache import BatchKVCache, CacheList, CacheMemory, KVCache

logger = logging.getLogger(__name__)


class _NotDeferrable(Exception):
    pass


class _RowPrefix:
    """The first ``n`` tokens of one row of an append-only K/V cache."""

    __slots__ = ("n",)

    def __init__(self, n: int):
        self.n = n


def _defer_entry(c: Any, idx: int, n: int, eval_targets: List[mx.array]) -> Any:
    if isinstance(c, CacheList):
        return CacheList(*(_defer_entry(ch, idx, n, eval_targets) for ch in c.caches))
    # Exact types: positions before the boundary are never rewritten.
    if type(c) is BatchKVCache and c.keys is not None:
        if c._idx - int(c.left_padding[idx].item()) != n:
            raise _NotDeferrable
        return _RowPrefix(n)
    if type(c) is KVCache and c.keys is not None:
        if idx != 0 or c.offset != n:
            raise _NotDeferrable
        return _RowPrefix(n)
    # A sliding window or recurrent state moves on: copy it now (it is small).
    extract = getattr(c, "extract", None)
    row = (
        extract(idx)
        if callable(extract)
        else clone_cache_entry(c, min_capacity_tokens=None, eval_targets=eval_targets)
    )
    if row is None:
        raise _NotDeferrable
    eval_targets.extend(a for a in _arrays(row.state))
    return row


def _arrays(tree: Any):
    if isinstance(tree, mx.array):
        yield tree
    elif isinstance(tree, (list, tuple)):
        for t in tree:
            yield from _arrays(t)


def _prefix_bytes(live: Any, tmpl: Any) -> int:
    if isinstance(tmpl, CacheList):
        return sum(_prefix_bytes(l, t) for l, t in zip(live.caches, tmpl.caches))
    if not isinstance(tmpl, _RowPrefix) or live.keys is None:
        return 0
    per_token = (live.keys[0, :, 0].nbytes + live.values[0, :, 0].nbytes)
    return per_token * tmpl.n


def _finish_entry(live: Any, tmpl: Any, idx: int, release: bool) -> Any:
    if isinstance(tmpl, CacheList):
        if not isinstance(live, CacheList) or len(live.caches) != len(tmpl.caches):
            raise _NotDeferrable
        return CacheList(
            *(_finish_entry(l, t, idx, release) for l, t in zip(live.caches, tmpl.caches))
        )
    if not isinstance(tmpl, _RowPrefix):
        return tmpl
    # The row may have been merged into a batch (or the batch filtered) since.
    if type(live) is BatchKVCache:
        start, end = int(live.left_padding[idx].item()), live._idx
    elif type(live) is KVCache and idx == 0:
        start, end = 0, live.offset
    else:
        raise _NotDeferrable
    if live.keys is None or start < 0 or start + tmpl.n > end:
        raise _NotDeferrable
    out = KVCache()
    out.keys = mx.contiguous(live.keys[idx : idx + 1, :, start : start + tmpl.n])
    out.values = mx.contiguous(live.values[idx : idx + 1, :, start : start + tmpl.n])
    out.offset = tmpl.n
    mx.eval(out.keys, out.values)
    if release:
        # Layer by layer: the snapshot replaces the live cache as it is built.
        live.keys = live.values = None
    return out


class DeferredCheckpoint:
    """A prefill checkpoint whose dense K/V is still the request's (APC_SINGLE_COPY).

    Only the state that moves on (a sliding window, recurrent or convolution
    state) is copied at the boundary. The append-only K/V rows are read from
    the live cache when the row leaves the batch; when it is the batch's last
    row (``release``), each layer's buffers are freed as its snapshot is made,
    so the turn never holds a second copy of the context.
    """

    def __init__(self, coordinator, token_ids, extra_hash, template):
        self.coordinator = coordinator
        self.token_ids = list(token_ids)
        self.extra_hash = extra_hash
        self.template = template

    def finish(self, prompt_cache: Sequence[Any], batch_idx: int, *, release: bool) -> bool:
        from .apc import apc_trace

        manager = self.coordinator.manager
        continues = manager.extends_resident_checkpoint(self.token_ids, self.extra_hash)
        if len(prompt_cache) != len(self.template):
            return False
        if not release:
            # The batch keeps its rows: this snapshot is a copy.
            size = sum(_prefix_bytes(l, t) for l, t in zip(prompt_cache, self.template))
            if not manager._make_room(size, prefill_reserve=not continues):
                apc_trace("store", mode="deferred", ok=False, reason="room",
                          token_len=len(self.token_ids), size_mb=size >> 20)
                with manager.lock:
                    manager.stats.memory_skips += 1
                return False
        try:
            snapshot = [
                _finish_entry(live, tmpl, batch_idx, release)
                for live, tmpl in zip(prompt_cache, self.template)
            ]
        except _NotDeferrable:
            return False
        apc_trace("store", mode="deferred", release=release, token_len=len(self.token_ids))
        return manager.store_exact_cache(
            self.token_ids,
            snapshot,
            extra_hash=self.extra_hash,
            owned=True,
            continues=continues,
            allocated=True,
        )


def finish_deferred_checkpoints(
    deferred: Sequence[DeferredCheckpoint],
    prompt_cache: Sequence[Any],
    batch_idx: int,
    *,
    release: bool,
) -> None:
    """Store a leaving row's deferred checkpoints, the longest last (it may take
    the row's buffers). With APC_EVICT_SUPERSEDED the longest would evict the
    others at once, so only it is stored."""
    ordered = sorted(deferred, key=lambda d: len(d.token_ids))
    if not ordered:
        return
    if ordered[-1].coordinator.manager.evict_superseded:
        ordered = ordered[-1:]
    for i, d in enumerate(ordered):
        try:
            d.finish(prompt_cache, batch_idx, release=release and i == len(ordered) - 1)
        except Exception as e:  # a lost checkpoint must not fail the request
            logger.warning("APC deferred checkpoint store failed: %s", e)


class PrefillMemoryPlan:
    """Per-manager budgets from cache layouts or conservative opaque bounds.

    Empty caches use the device reserve until dimensions are known.
    """

    def __init__(self):
        self.components: List[CacheMemory] = []
        self.lengths: List[int] = []
        self.prefix_lengths: List[int] = []
        self.chunk_size: Optional[int] = None
        self.extra_reserve_bytes = 0

    def prepare(
        self, lengths, *, chunk_size=None, prefix_lengths=None, extra_reserve_bytes=0
    ) -> int:
        self.extra_reserve_bytes = max(0, int(extra_reserve_bytes))
        self.lengths = [max(0, int(n)) for n in lengths]
        self.prefix_lengths = (
            list(prefix_lengths)
            if prefix_lengths is not None
            else [0] * len(self.lengths)
        )
        self.chunk_size = chunk_size if chunk_size and chunk_size > 0 else None
        return self.reserve_bytes()

    def reserve_bytes(self, live_bytes: int = 0) -> int:
        if not self.lengths or not max(self.lengths):
            return 0
        # Prefixes and suffixes are padded separately; cold prefixes are zero.
        capacity = max(self.prefix_lengths) + max(
            n - p for n, p in zip(self.lengths, self.prefix_lengths)
        )
        row_bytes = sum(c.footprint(capacity, self.chunk_size) for c in self.components)
        # A restored prefix is allocated by its restore (``restore_bytes``, at the
        # prompt's capacity), not grown by prefill, so only the suffix needs the
        # growth reserve. Otherwise a short follow-up on a long conversation
        # reserves twice the whole cache again on top of the restored copy, and
        # near the window it is refused outright.
        restored = min(self.prefix_lengths)
        if restored:
            row_bytes -= sum(
                c.footprint(restored, self.chunk_size) for c in self.components
            )
        return self.extra_reserve_bytes + max(
            0, 2 * len(self.lengths) * max(0, row_bytes) - live_bytes
        )

    def observe(self, components, *, live_bytes: int) -> int:
        previous = self.components
        self.components = list(components)
        if len(previous) == len(self.components):
            for i, (old, new) in enumerate(zip(previous, self.components)):
                if not new.source_bytes:
                    self.components[i] = old
                elif old.fallback and new.fallback:
                    self.components[i] = replace(
                        new,
                        bytes_per_token=max(old.bytes_per_token, new.bytes_per_token),
                    )
        return self.reserve_bytes(live_bytes)

    def observe_cache(
        self, prompt_cache, token_count, *, batch_size=1, live_bytes=None
    ) -> int:
        """Update dimensions and return the reserve beyond live buffers."""
        from .apc import _cache_nbytes

        if live_bytes is None:
            live_bytes = _cache_nbytes(prompt_cache)
        return self.observe(
            cache_memory_components(prompt_cache, token_count, batch_size=batch_size),
            live_bytes=live_bytes,
        )

    def observe_kv(self, keys, values, *, live_bytes: int) -> int:
        # Raw float KV slabs expose growth through their token axis.
        return self.observe(
            (
                CacheMemory(
                    source_bytes=k.nbytes + v.nbytes,
                    bytes_per_token=(k.nbytes + v.nbytes) / k.shape[2],
                )
                for k, v in zip(keys, values)
                if k.shape[2]
            ),
            live_bytes=live_bytes,
        )

    def restore_bytes(self, prompt_cache, token_count, capacity) -> int:
        components = cache_memory_components(prompt_cache, token_count)
        return sum(
            max(c.source_bytes, c.footprint(capacity, self.chunk_size))
            for c in components
        )


class APCCoordinator:
    """Coordinate prefix reuse across every cache group in a model.

    Native dense K/V layouts use the block pool.  Windowed, recurrent,
    composite and custom layouts use restorable checkpoints at the same token
    boundary.  The distinction is deliberately private to this class.
    """

    def __init__(self, manager: Any, model: Any):
        self.manager = manager
        self.model = model
        self.plan: PrefixCachePlan = build_prefix_cache_plan(model)

    def prepare_prefill(
        self,
        prompt_lengths,
        *,
        prefill_step_size=None,
        prefix_lengths=None,
        extra_reserve_bytes=0,
    ) -> bool:
        if self.enabled:
            lengths = (
                [prompt_lengths] if isinstance(prompt_lengths, int) else prompt_lengths
            )
            reserve = self.manager.memory_plan.prepare(
                lengths,
                chunk_size=prefill_step_size,
                prefix_lengths=prefix_lengths,
                extra_reserve_bytes=extra_reserve_bytes,
            )
            return self.manager.prepare_prefill(reserve)
        return True

    def prefix_hint(
        self, token_ids, *, extra_hash: int = 0, min_prefix_tokens: int = 0
    ) -> int:
        """Resident checkpoint length a lookup of ``token_ids`` would restore,
        found without copying it."""
        hint = getattr(self.manager, "exact_prefix_hint", None)
        if not self.enabled or hint is None:
            return 0
        return hint(
            token_ids, extra_hash=extra_hash, min_prefix_tokens=min_prefix_tokens
        )

    def finish_prefill(self) -> None:
        """Release the transient plan after admission is deferred or completed."""
        if self.enabled:
            self.manager.memory_plan.prepare([])
            self.manager._prefill_reserve_bytes = 0

    def observe_cache(self, prompt_cache, token_count, *, batch_size=1) -> None:
        """Update runtime dimensions independently of snapshot persistence."""
        reserve = self.manager.memory_plan.observe_cache(
            prompt_cache, token_count, batch_size=batch_size
        )
        self.manager.prepare_prefill(reserve)

    @property
    def enabled(self) -> bool:
        return self.manager is not None and self.plan.restorable

    @property
    def strategy(self) -> Optional[str]:
        return self.plan.strategy if self.enabled else None

    @property
    def is_checkpoint(self) -> bool:
        return self.strategy == "checkpoint"

    @property
    def legacy_mode(self) -> Optional[str]:
        return self.plan.legacy_mode if self.enabled else None

    def fresh_cache(self) -> List[Any]:
        language_model = getattr(self.model, "language_model", self.model)
        make_cache = getattr(language_model, "make_cache", None) or getattr(
            self.model, "make_cache", None
        )
        if callable(make_cache):
            return list(make_cache())
        from .models.cache import make_prompt_cache

        return list(make_prompt_cache(language_model))

    def lookup(
        self,
        token_ids: Sequence[int],
        *,
        extra_hash: int,
        safe_lookup_min: int,
        suffix_is_text_only: Callable[[int], bool],
        prefix_has_media: Callable[[int], bool],
        defer_on_pressure: bool = False,
    ) -> Optional[dict]:
        if not self.enabled:
            return None
        from .apc import apc_lookup_plan

        hit = apc_lookup_plan(
            self.manager,
            token_ids,
            extra_hash=extra_hash,
            apc_mode=self.legacy_mode,
            safe_lookup_min=safe_lookup_min,
            suffix_is_text_only=suffix_is_text_only,
            prefix_has_media=prefix_has_media,
            defer_on_pressure=defer_on_pressure,
        )
        if hit is not None:
            hit["cache_plan"] = self.plan
        return hit

    def checkpoint_len(
        self, token_ids: Sequence[int], media_token_ids: set[int]
    ) -> int:
        """Reusable checkpoint before the final guard token(s)."""
        if not self.enabled or not self.is_checkpoint:
            return 0
        from .apc import adjust_prefix_to_text_suffix_boundary

        return adjust_prefix_to_text_suffix_boundary(
            token_ids,
            len(token_ids) - self.manager.exact_cache_guard_tokens,
            media_token_ids,
            max_prefix_tokens=len(token_ids) - 1,
        )

    def checkpoint_lengths(
        self,
        token_ids: Sequence[int],
        media_token_ids: set[int],
        *,
        extra_hash: int = 0,
        stable_prefix_len: int = 0,
    ) -> List[int]:
        """Bounded intermediate states plus the final conversation checkpoint.

        Stateful caches cannot roll back the final snapshot to a divergence.
        Capture earlier states while prefilling, aligned across requests. Limit
        captures to the resident entry budget (two for a disk-only manager),
        rather than copying an ever-growing cache at every prefill chunk.
        """
        final = self.checkpoint_len(token_ids, media_token_ids)
        if final <= 0:
            return []
        interval = self.manager.checkpoint_interval_tokens
        budget = self.manager._exact_cache_max or (2 if self.manager.disk else 1)
        if interval <= 0 or budget <= 1:
            return [final]
        from .apc import adjust_prefix_to_text_suffix_boundary

        block_size = self.manager.block_size
        interval = ((interval + block_size - 1) // block_size) * block_size
        last = ((final - 1) // interval) * interval
        first = max(interval, last - (budget - 2) * interval)
        lengths = {final}
        # Prefer the observed stable prefix over positions inside a changing
        # suffix. The final checkpoint still serves exact prompt extensions.
        stable = self.manager.checkpoint_prefix_length(token_ids, extra_hash=extra_hash)
        candidates = [stable_prefix_len, stable] + list(
            range(last, first - 1, -interval)
        )
        for boundary in candidates:
            if boundary < self.manager.exact_cache_min_tokens:
                continue
            boundary = adjust_prefix_to_text_suffix_boundary(
                token_ids, boundary, media_token_ids, max_prefix_tokens=final
            )
            if self.manager.exact_cache_min_tokens <= boundary < final:
                lengths.add(boundary)
            if len(lengths) >= budget:
                break
        return sorted(lengths)

    def merge_rows(
        self,
        picks: Sequence[Optional[dict]],
        prefix_lens: Sequence[int],
        *,
        kv_quant_config: Optional[dict] = None,
    ) -> Tuple[Optional[List[Any]], int]:
        """Materialize a mixed batch while preserving the caller's cache rows."""
        source = (
            picks[0]["warm_cache"]
            if self.is_checkpoint and len(picks) == 1 and picks[0] is not None
            else None
        )
        if self.is_checkpoint:
            picks = [
                (
                    {**pick, "warm_cache": list(pick["warm_cache"])}
                    if pick is not None
                    else None
                )
                for pick in picks
            ]
        caches, prefix_len = self._merge_and_release_cache_rows(
            picks, prefix_lens, kv_quant_config=kv_quant_config
        )
        if caches is not None and source is not None:
            from .apc import _detach_borrowed_cache_entries

            caches = _detach_borrowed_cache_entries(caches, source)
        return caches, prefix_len if caches is not None else 0

    def _merge_and_release_cache_rows(
        self,
        picks: Sequence[Optional[dict]],
        prefix_lens: Sequence[int],
        *,
        kv_quant_config: Optional[dict] = None,
    ) -> Tuple[Optional[List[Any]], int]:
        """Merge scheduler-owned rows; discard picks after success or failure.

        Exact-cache input entries are released layer by layer. Block leases
        retain their existing lifecycle in the scheduler.
        """
        from .apc import _merge_and_release_cache_rows, make_warm_batch_kv_cache_multi

        memory = self.manager.memory_plan
        if len(prefix_lens) == len(memory.lengths):
            if not self.prepare_prefill(
                memory.lengths,
                prefill_step_size=memory.chunk_size,
                prefix_lengths=prefix_lens,
                extra_reserve_bytes=memory.extra_reserve_bytes,
            ):
                from .apc import APCAdmissionError

                for pick in picks:  # the request waits: its checkpoints go back
                    if pick is not None and pick.get("warm_cache") is not None:
                        self.manager.return_taken(
                            pick["full_input_ids"][: pick["prefix_len"]],
                            pick["warm_cache"],
                            pick.get("extra_hash", 0),
                        )
                raise APCAdmissionError("APC batch merge deferred by memory admission")

        if self.is_checkpoint:
            row_caches = [
                pick["warm_cache"] if pick is not None else self.fresh_cache()
                for pick in picks
            ]
            caches, max_prefix = _merge_and_release_cache_rows(
                row_caches,
                prefix_lens,
                kv_quant_config=kv_quant_config,
            )
        else:
            caches, max_prefix = make_warm_batch_kv_cache_multi(
                list(picks),
                num_layers=len(self.plan.components),
                kv_quant_config=kv_quant_config,
            )
        if caches is not None:
            with self.manager.lock:
                self.manager.stats.restored_tokens += sum(prefix_lens)
        return caches, max_prefix

    def materialize_single(
        self,
        hit: dict,
        *,
        min_capacity_tokens: int,
        kv_quant_config: Optional[dict] = None,
    ) -> List[Any]:
        warm_cache = hit.get("warm_cache")
        if warm_cache is None:
            from .apc import make_warm_kv_cache

            warm_cache = make_warm_kv_cache(
                hit.get("matched_blocks", []),
                min_capacity_tokens=min_capacity_tokens,
                kv_quant_config=kv_quant_config,
            )
        if warm_cache:
            with self.manager.lock:
                self.manager.stats.restored_tokens += hit["prefix_len"]
        return warm_cache

    def defer_checkpoint(
        self,
        token_ids: Sequence[int],
        prompt_cache: Sequence[Any],
        *,
        extra_hash: int = 0,
        batch_idx: int = 0,
    ) -> Optional[DeferredCheckpoint]:
        """With APC_SINGLE_COPY, a checkpoint of the first ``len(token_ids)``
        tokens of a row to store when the row leaves (``DeferredCheckpoint``),
        or None: store it now with ``store_checkpoint``."""
        if not (self.enabled and self.is_checkpoint and self.manager.single_copy):
            return None
        if len(token_ids) < self.manager.exact_cache_min_tokens:
            return None
        if not build_prefix_cache_plan_from_caches(prompt_cache).restorable:
            return None
        eval_targets: List[mx.array] = []
        try:
            template = [
                _defer_entry(c, batch_idx, len(token_ids), eval_targets)
                for c in prompt_cache
            ]
        except _NotDeferrable:
            return None
        if eval_targets:
            mx.eval(eval_targets)
        return DeferredCheckpoint(self, token_ids, extra_hash, template)

    def store_checkpoint(
        self,
        token_ids: Sequence[int],
        prompt_cache: Sequence[Any],
        *,
        extra_hash: int = 0,
        batch_idx: Optional[int] = None,
    ) -> bool:
        if not self.enabled or not self.is_checkpoint:
            return False
        from .apc import (
            _cache_nbytes,
            _prompt_cache_is_batch_shaped,
            apc_trace,
            snapshot_prompt_cache_row,
        )

        # Decide before making room evicts the checkpoint this one continues.
        continues = self.manager.extends_resident_checkpoint(token_ids, extra_hash)
        # Batch extraction can itself allocate a full row before store_exact_cache
        # decides whether it can afford another retained snapshot.
        if _prompt_cache_is_batch_shaped(prompt_cache):
            if self.manager.disk is not None:
                self.manager.disk.flush()
            if not self.manager._make_room(
                _cache_nbytes(prompt_cache), prefill_reserve=not continues
            ):
                apc_trace(
                    "store",
                    mode="exact",
                    ok=False,
                    reason="room-before-snapshot",
                    token_len=len(token_ids),
                    continues=continues,
                    size_mb=_cache_nbytes(prompt_cache) >> 20,
                    headroom_mb=self.manager._memory_headroom() >> 20,
                    resident_mb=self.manager.resident_bytes() >> 20,
                )
                with self.manager.lock:
                    self.manager.stats.memory_skips += 1
                return False
        snapshot = snapshot_prompt_cache_row(prompt_cache, batch_idx or 0, clone=False)
        if snapshot is None:
            return False
        # A row extracted from batch caches is already a contiguous private copy.
        return self.manager.store_exact_cache(
            token_ids,
            snapshot,
            extra_hash=extra_hash,
            owned=_prompt_cache_is_batch_shaped(prompt_cache),
            continues=continues,
        )

    def commit(
        self,
        prompt_cache: Sequence[Any],
        token_ids: Sequence[int],
        *,
        batch_idx: Optional[int] = None,
        extra_hash: int = 0,
        skip_first_n_tokens: int = 0,
        blocks_in_use: Sequence[Any] = (),
    ) -> bool:
        """Store one completed prefix and release any block leases."""
        if not self.enabled:
            return False
        if self.is_checkpoint:
            try:
                return self.store_checkpoint(
                    token_ids,
                    prompt_cache,
                    batch_idx=batch_idx,
                    extra_hash=extra_hash,
                )
            finally:
                self.manager.release(blocks_in_use)

        from .apc import commit_prefix_blocks

        commit_prefix_blocks(
            self.manager,
            list(prompt_cache),
            token_ids,
            batch_idx=batch_idx,
            extra_hash=extra_hash,
            skip_first_n_tokens=skip_first_n_tokens,
            blocks_in_use=blocks_in_use,
        )
        return True

    def release_hit(self, hit: Optional[dict]) -> None:
        if hit is not None and self.manager is not None:
            self.manager.release(hit.get("matched_blocks", ()))
