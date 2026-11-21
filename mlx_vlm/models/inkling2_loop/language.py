# Copyright 2026 the HuggingFace Team. All rights reserved.
# Adapted for MLX from Transformers' Inkling2Loop implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mlx.core as mx
import mlx.nn as nn

from ..inkling.language import _cache_padding_mask
from ..inkling2 import fused
from ..inkling2.language import AttentionResidual, DecoderLayer
from ..inkling2.language import LanguageModel as Inkling2LanguageModel
from ..inkling2.language import _attention_residual
from .config import TextConfig


@mx.compile
def _message_gate(hidden, message, down, forget, gain, eps):
    hidden = hidden.astype(mx.float32)
    message = message.astype(mx.float32)
    gate_input = mx.concatenate(
        [mx.fast.rms_norm(hidden, None, eps), mx.fast.rms_norm(message, None, eps)],
        axis=-1,
    )
    gate_input = (gate_input.astype(down.dtype) @ down.T).astype(mx.float32)
    return (1 - mx.tanh(gate_input @ forget.T)) * hidden + mx.tanh(
        gate_input @ gain.T
    ) * message


class MessageGate(nn.Module):
    """(1 - forget) * hidden + gain * message, gated from both normalized inputs."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.eps = config.rms_norm_eps
        rank = config.loop_gate_rank
        self.down_proj = nn.Linear(2 * config.hidden_size, rank, bias=False)
        self.forget_gate_proj = nn.Linear(rank, config.hidden_size, bias=False)
        self.message_gate_proj = nn.Linear(rank, config.hidden_size, bias=False)

    def __call__(self, hidden, message):
        projections = (self.down_proj, self.forget_gate_proj, self.message_gate_proj)
        if all(type(p) is nn.Linear and "bias" not in p for p in projections):
            # 96 gates run per token: one compiled graph each, and the FP32 copies
            # of the two rank-64 projections made once instead of every call.
            forget, gain = self._fp32_weights()
            return _message_gate(
                hidden, message, self.down_proj.weight, forget, gain, self.eps
            )
        hidden = hidden.astype(mx.float32)
        message = message.astype(mx.float32)
        gate_input = mx.concatenate(
            [
                mx.fast.rms_norm(hidden, None, self.eps),
                mx.fast.rms_norm(message, None, self.eps),
            ],
            axis=-1,
        )
        gate_input = self.down_proj(
            gate_input.astype(self.down_proj.weight.dtype)
        ).astype(mx.float32)
        forget = mx.tanh(
            gate_input @ self.forget_gate_proj.weight.astype(mx.float32).T
        )
        gain = mx.tanh(gate_input @ self.message_gate_proj.weight.astype(mx.float32).T)
        return (1 - forget) * hidden + gain * message

    def _fp32_weights(self):
        forget, gain = self.forget_gate_proj.weight, self.message_gate_proj.weight
        cached = getattr(self, "_fp32_cache", None)
        if cached is None or cached[0] is not forget or cached[1] is not gain:
            cached = (forget, gain, forget.astype(mx.float32), gain.astype(mx.float32))
            # A plain attribute: not a parameter, so never saved or loaded.
            object.__setattr__(self, "_fp32_cache", cached)
        return cached[2], cached[3]


class MessageAttentionResidual(nn.Module):
    """Reads the query, key, value and MLP messages of each loop layer from the
    previous pass: one learned query per message over that pass's block states."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.query = mx.zeros((config.num_loop_layers, 4, config.hidden_size))
        self.normalize_values = config.attn_res_normalize_values

    def __call__(self, blocks, partial, loop_layer_idx):
        states = mx.stack([*blocks, partial], axis=-2)
        return _attention_residual(
            states, self.query[loop_layer_idx], self.normalize_values
        )


class LoopDecoderLayer(DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.q_message_gate = MessageGate(config)
        self.k_message_gate = MessageGate(config)
        self.v_message_gate = MessageGate(config)
        self.mlp_message_gate = MessageGate(config)

    def __call__(self, x, blocks, cache=None, messages=None):
        if messages is None:
            return super().__call__(x, blocks, cache)
        conv = cache[1] if cache is not None else None
        mask = _cache_padding_mask(conv, x.shape[1])
        mixed = self.attention_input(blocks, x)
        q_msg, k_msg, v_msg, mlp_msg = (messages[..., i, :] for i in range(4))
        if self.is_sliding:
            # Sliding layers share one attention input but gate it per projection.
            shared = mixed[..., 0, :]
            parts = (shared, shared, shared)
            norms = (self.input_layernorm,) * 3
        else:
            parts = tuple(mixed[..., i, :] for i in range(3))
            norms = (
                self.q_input_layernorm,
                self.k_input_layernorm,
                self.v_input_layernorm,
            )
        parts = [
            norm(gate(part, msg)).astype(x.dtype)
            for norm, gate, part, msg in zip(
                norms,
                (self.q_message_gate, self.k_message_gate, self.v_message_gate),
                parts,
                (q_msg, k_msg, v_msg),
            )
        ]
        if self.is_sliding:
            attn = self.self_attn(
                parts[0],
                cache=cache,
                conv_mask=mask,
                key_input=parts[1],
                value_input=parts[2],
            )
        else:
            attn = self.self_attn(*parts, cache=cache, conv_mask=mask)
        residual = self.attn_sconv(attn, conv, mask).astype(mx.float32)
        if self.is_block_boundary:
            blocks.append(x)
        else:
            residual = residual + x.astype(mx.float32)
        hidden = self.mlp_input(blocks, residual)[..., 0, :]
        hidden = self.mlp_message_gate(hidden, mlp_msg)
        hidden = self.mlp(self.post_attention_layernorm(hidden).astype(x.dtype))
        out = (residual + self.mlp_sconv(hidden, conv, mask).astype(mx.float32)).astype(
            x.dtype
        )
        if conv is not None:
            conv.advance(x.shape[1])
        return out


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        start, end = config.loop_layer_index_start, config.loop_layer_index_end
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [
            (LoopDecoderLayer if start <= i < end else DecoderLayer)(config, i)
            for i in range(config.num_hidden_layers)
        ]
        self.message_input = MessageAttentionResidual(config)
        self.final_input = AttentionResidual(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, inputs):
        return self.embed_norm(self.embed_tokens(inputs))

    def schedule(self):
        """(layer index, cache index, pass) for every layer application."""
        cfg = self.config
        start, end = cfg.loop_layer_index_start, cfg.loop_layer_index_end
        out = [(i, i, 0) for i in range(start)]
        for p in range(cfg.loop_num_passes):
            for i in range(start, end):
                cache_idx = i if p == 0 else (
                    cfg.num_hidden_layers + (p - 1) * cfg.num_loop_layers + i - start
                )
                out.append((i, cache_idx, p))
        out += [(i, i, 0) for i in range(end, cfg.num_hidden_layers)]
        return out

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        skip_final_norm=False,
        capture_layer_ids=None,
        speculative_verify=False,
    ):
        cfg = self.config
        total = cfg.total_num_hidden_layers
        if capture_layer_ids is not None and (
            len(capture_layer_ids) != len(set(capture_layer_ids))
            or any(type(i) is not int or not 0 <= i <= total for i in capture_layer_ids)
        ):
            raise ValueError("Invalid Inkling 2 capture_layer_ids")
        requested = set(capture_layer_ids or [])
        captures = {}
        x = self.embed(inputs) if input_embeddings is None else input_embeddings
        if capture_layer_ids is None and not speculative_verify and fused.supported(self, x, cache):
            # One token a row: each layer visit in a few kernels instead of ~125 MLX operations.
            h = fused.decode_step(self, x, cache)
            return h if skip_final_norm else self.norm(h).astype(x.dtype)
        start = cfg.loop_layer_index_start
        blocks = []
        pre_loop = None
        previous = None
        for step, (i, cache_idx, p) in enumerate(self.schedule()):
            layer = self.layers[i]
            if i == start:
                if p == 0:
                    pre_loop = (x, list(blocks))
                else:
                    # Restart from the pre-loop state; read messages from the
                    # blocks the previous pass added after its first block.
                    previous = (blocks[len(pre_loop[1]) + 1 :], x)
                    x, blocks = pre_loop[0], list(pre_loop[1])
            if step in requested:
                captures[step] = x
            c = cache[cache_idx] if cache is not None else None
            if p > 0:
                messages = self.message_input(previous[0], previous[1], i - start)
                x = layer(x, blocks, c, messages)
            else:
                x = layer(x, blocks, c)
            if speculative_verify and (step + 1) % 4 == 0:
                mx.async_eval(x)
        h = self.final_input(blocks, x)[..., 0, :]
        normalized = self.norm(h).astype(x.dtype)
        if total in requested:
            captures[total] = normalized
        result = h if skip_final_norm else normalized
        if capture_layer_ids is not None:
            return result, [captures[i] for i in capture_layer_ids]
        return result


class LanguageModel(Inkling2LanguageModel):
    def __init__(self, config: TextConfig):
        nn.Module.__init__(self)
        self.config = config
        self.model_type = config.model_type
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.sharding_group = None

    def make_cache(self):
        caches = super().make_cache()
        cfg = self.config
        loop = caches[cfg.loop_layer_index_start : cfg.loop_layer_index_end]
        for _ in range(cfg.loop_num_passes - 1):
            for layer in self.model.layers[
                cfg.loop_layer_index_start : cfg.loop_layer_index_end
            ]:
                caches.append(type(loop[0])(*self._layer_cache(layer)))
        return caches

    def _layer_cache(self, layer):
        from ..cache import ArraysCache, KVCache, RotatingKVCache

        return (
            (
                RotatingKVCache(max_size=self.config.sliding_window_size, keep=0)
                if layer.is_sliding
                else KVCache()
            ),
            ArraysCache(4),
        )
