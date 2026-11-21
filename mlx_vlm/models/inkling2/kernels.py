"""Inference kernels that retain Inkling 2's FP32 intermediate arithmetic."""

import math
from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=1)
def _short_convolution():
    return mx.fast.metal_kernel(
        name="inkling2_virtual_short_convolution",
        input_names=["state", "weight", "x", "mask", "divisor"],
        output_names=["out"],
        ensure_row_contiguous=False,
        compile_options={"math_mode": "safe"},
        source=r"""
        uint i = thread_position_in_grid.x;
        int B = x_shape[0], L = x_shape[1], C = x_shape[2];
        if (i >= B * L * C) return;
        int c = i % C, t = (i / C) % L, b = i / (C * L);
        float value = 0;
        for (int j = 0; j < W; ++j) {
            int p = t + j - W + 1;
            float xv = p < 0
                ? float(state[b * state_strides[0] + (p + W - 1) * state_strides[1]
                              + c * state_strides[2]])
                : ((HAS_MASK && !mask[b * mask_strides[0] + p * mask_strides[1]])
                    ? 0.0f
                    : float(x[b * x_strides[0] + p * x_strides[1] + c * x_strides[2]]));
            float w = weight[c * weight_strides[0] + j * weight_strides[1]];
            value = value + xv * w;
        }
        out[i] = T((value + float(x[b * x_strides[0] + t * x_strides[1]
                                   + c * x_strides[2]])) / divisor);
        """,
    )


def short_convolution(state, weight, x, mask, width):
    """Read the virtual state/input concatenation without an FP32 input copy."""
    return _short_convolution()(
        inputs=[
            state,
            weight,
            x,
            mask if mask is not None else mx.zeros((1, 1), mx.bool_),
            mx.array(math.sqrt(width + 1), mx.float32),
        ],
        template=[("T", x.dtype), ("W", width), ("HAS_MASK", mask is not None)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


# Rows of the head per simdgroup, and simdgroups per threadgroup.
_HEAD_ROWS = 4
_HEAD_SIMDGROUPS = 8


@lru_cache(maxsize=1)
def _head_logits():
    return mx.fast.metal_kernel(
        name="inkling2_bf16_head_fp32_logits",
        input_names=["h", "w"],
        output_names=["out"],
        source=r"""
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int row0 = (int(threadgroup_position_in_grid.x) * SG + int(sg)) * R;
        if (row0 >= N) return;
        float acc[R][M];
        for (int r = 0; r < R; ++r)
            for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
        for (int c = lane; c < K / 8; c += 32) {
            const int k = c * 8;
            float wv[R][8];
            for (int r = 0; r < R; ++r) {
                const int row = min(row0 + r, N - 1);
                auto wp = (const device vec<T, 4>*)(w + size_t(row) * K + k);
                const vec<T, 4> a = wp[0], b = wp[1];
                for (int j = 0; j < 4; ++j) {
                    wv[r][j] = float(a[j]);
                    wv[r][j + 4] = float(b[j]);
                }
            }
            for (int m = 0; m < M; ++m) {
                auto hp = (const device float4*)(h + m * K + k);
                const float4 x0 = hp[0], x1 = hp[1];
                for (int r = 0; r < R; ++r) {
                    acc[r][m] += wv[r][0] * x0.x + wv[r][1] * x0.y + wv[r][2] * x0.z
                        + wv[r][3] * x0.w + wv[r][4] * x1.x + wv[r][5] * x1.y
                        + wv[r][6] * x1.z + wv[r][7] * x1.w;
                }
            }
        }
        for (int r = 0; r < R; ++r) {
            for (int m = 0; m < M; ++m) {
                const float total = simd_sum(acc[r][m]);
                if (lane == 0 && row0 + r < N) out[size_t(m) * N + row0 + r] = total;
            }
        }
        """,
    )


def head_logits(h, weight):
    """FP32 logits of FP32 rows h (M, K) against a BF16 head (N, K), reading
    the BF16 weights directly instead of an FP32 copy of the head."""
    m, k = h.shape
    n = weight.shape[0]
    rows = _HEAD_ROWS * _HEAD_SIMDGROUPS
    return _head_logits()(
        inputs=[mx.contiguous(h.astype(mx.float32)), weight],
        template=[
            ("T", weight.dtype),
            ("M", m),
            ("K", k),
            ("N", n),
            ("R", _HEAD_ROWS),
            ("SG", _HEAD_SIMDGROUPS),
        ],
        grid=((n + rows - 1) // rows * 32 * _HEAD_SIMDGROUPS, 1, 1),
        threadgroup=(32 * _HEAD_SIMDGROUPS, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[mx.float32],
    )[0]


# Single-query decode attention over a long chronological K/V cache, split over
# key blocks (flash-decoding). One threadgroup reads each key block of one KV
# head once for all of that head's query heads (GQA), instead of once per query
# head; the relative bias of the newest REL keys and the sink are applied here.
_DECODE_BLOCK = 512
_DECODE_SIMDGROUPS = 8


@lru_cache(maxsize=1)
def _decode_partials():
    return mx.fast.metal_kernel(
        name="inkling2_decode_attention_partials",
        input_names=["q", "k", "v", "rel", "key_start", "sizes", "scale"],
        output_names=["part_max", "part_sum", "part_acc"],
        ensure_row_contiguous=False,
        source=r"""
        constexpr int G = HQ / HKV;
        constexpr int E = D / 32;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const int block = threadgroup_position_in_grid.x;
        const int g = threadgroup_position_in_grid.y;
        const int S = sizes[0];
        const int start = block * BS;
        const int end = min(start + BS, S);
        // Keys before key_start are a batch row's left padding.
        const int first = max(start, key_start[0]);
        const float sc = scale;

        float qv[G][E];
        for (int h = 0; h < G; ++h)
            for (int e = 0; e < E; ++e)
                qv[h][e] = float(q[(g * G + h) * D + lane * E + e]) * sc;
        float m[G], l[G], acc[G][E];
        for (int h = 0; h < G; ++h) {
            m[h] = -INFINITY;
            l[h] = 0.0f;
            for (int e = 0; e < E; ++e) acc[h][e] = 0.0f;
        }
        const device T* k_head = k + g * k_strides[1];
        const device T* v_head = v + g * v_strides[1];
        for (int j = first + int(sg); j < end; j += SG) {
            const device T* kp = k_head + size_t(j) * k_strides[2] + lane * E;
            const device T* vp = v_head + size_t(j) * v_strides[2] + lane * E;
            float kv[E], vv[E];
            for (int e = 0; e < E; ++e) {
                kv[e] = float(kp[e]);
                vv[e] = float(vp[e]);
            }
            const int dist = S - 1 - j;
            for (int h = 0; h < G; ++h) {
                float s = 0.0f;
                for (int e = 0; e < E; ++e) s += qv[h][e] * kv[e];
                s = simd_sum(s);
                if (dist < REL) s += float(rel[(g * G + h) * REL + dist]);
                const float mn = max(m[h], s);
                const float f = metal::exp(m[h] - mn);
                const float p = metal::exp(s - mn);
                l[h] = l[h] * f + p;
                for (int e = 0; e < E; ++e) acc[h][e] = acc[h][e] * f + p * vv[e];
                m[h] = mn;
            }
        }

        threadgroup float tm[SG][G];
        threadgroup float tl[SG][G];
        threadgroup float ta[SG][G][D];
        for (int h = 0; h < G; ++h) {
            if (lane == 0) {
                tm[sg][h] = m[h];
                tl[sg][h] = l[h];
            }
            for (int e = 0; e < E; ++e) ta[sg][h][lane * E + e] = acc[h][e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread t combines head t / D, dimension t % D across the simdgroups.
        for (uint t = thread_position_in_threadgroup.x; t < uint(G * D); t += SG * 32) {
            const int h = t / D, d = t % D;
            float M = -INFINITY;
            for (int s = 0; s < SG; ++s) M = max(M, tm[s][h]);
            float L = 0.0f, A = 0.0f;
            if (M > -INFINITY) {  // a block wholly inside the padding has no keys
                for (int s = 0; s < SG; ++s) {
                    const float w = metal::exp(tm[s][h] - M);
                    L += tl[s][h] * w;
                    A += ta[s][h][d] * w;
                }
            }
            const int row = block * HQ + g * G + h;
            part_acc[size_t(row) * D + d] = A;
            if (d == 0) {
                part_max[row] = M;
                part_sum[row] = L;
            }
        }
        """,
    )


@lru_cache(maxsize=1)
def _decode_combine():
    return mx.fast.metal_kernel(
        name="inkling2_decode_attention_combine",
        input_names=["part_max", "part_sum", "part_acc", "sinks", "sizes"],
        output_names=["out"],
        source=r"""
        const int h = threadgroup_position_in_grid.x;
        const int d = thread_position_in_threadgroup.x;
        const int nblocks = sizes[1];
        // The softmax starts from the sink logit, whose value row is zero.
        float M = float(sinks[h]);
        for (int b = 0; b < nblocks; ++b) M = max(M, part_max[b * HQ + h]);
        float L = metal::exp(float(sinks[h]) - M), A = 0.0f;
        for (int b = 0; b < nblocks; ++b) {
            const float w = metal::exp(part_max[b * HQ + h] - M);
            L += part_sum[b * HQ + h] * w;
            A += part_acc[size_t(b * HQ + h) * D + d] * w;
        }
        out[h * D + d] = T(A / L);
        """,
    )


def decode_attention(q, k, v, rel, sinks, *, scale, key_start=None):
    """One query of one sequence over a chronological cache.

    q: (1, HQ, 1, D); k, v: (1, HKV, S, D) with the query at the last key;
    rel: (HQ, R), the bias at backward distances below R; sinks: (HQ,);
    key_start: optional (1,) int array, the first valid key (left padding).
    Returns (1, HQ, 1, D) in q's dtype.
    """
    _, hq, _, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    nblocks = (s + _DECODE_BLOCK - 1) // _DECODE_BLOCK
    if key_start is None:
        key_start = mx.zeros((1,), mx.int32)
    common = [("T", q.dtype), ("D", d), ("HQ", hq), ("HKV", hkv)]
    part_max, part_sum, part_acc = _decode_partials()(
        inputs=[
            mx.contiguous(q.reshape(hq, d)),
            k,
            v,
            mx.contiguous(rel),
            key_start.astype(mx.int32),
            mx.array([s], dtype=mx.int32),
            mx.array(scale, dtype=mx.float32),
        ],
        template=[*common, ("REL", rel.shape[-1]), ("BS", _DECODE_BLOCK),
                  ("SG", _DECODE_SIMDGROUPS)],
        grid=(nblocks * 32 * _DECODE_SIMDGROUPS, hkv, 1),
        threadgroup=(32 * _DECODE_SIMDGROUPS, 1, 1),
        output_shapes=[(nblocks * hq,), (nblocks * hq,), (nblocks * hq, d)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    out = _decode_combine()(
        inputs=[part_max, part_sum, part_acc, sinks.astype(mx.float32),
                mx.array([s, nblocks], dtype=mx.int32)],
        template=[*common],
        grid=(hq * d, 1, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[(hq, d)],
        output_dtypes=[q.dtype],
    )[0]
    return out.reshape(1, hq, 1, d)


# The relative-attention kernel below is from ml-explore/mlx-lm pull request #2086
# (mlx_lm/models/inkling2/kernels.py, Copyright 2026 Apple Inc., MIT License).
# Causal flash attention with a relative position bias and a sink logit per
# query.
# Queries per threadgroup, 8 for each of its 4 simdgroups.
_BQ = 32

_TG = 128

# The online softmax of each row starts from its sink logit, so the zero-value
# sink needs no key.
_RELATIVE_ATTENTION_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    // q and k are clipped and v has unit RMS, so half holds them.
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            qv = half2(float(qp[0]), float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            if (key < S) {
                auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                kv[0] = kp[0];
                kv[1] = kp[1];
                vv[0] = vp[0];
                vv[1] = vp[1];
            }
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"""

_relative_attention_kernel = mx.fast.metal_kernel(
    name="inkling2_relative_attention",
    input_names=["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
    output_names=["out"],
    source=_RELATIVE_ATTENTION_SOURCE,
    ensure_row_contiguous=False,
)


def relative_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
) -> mx.array:
    """q: (B, Hq, L, D), k and v: (B, Hkv, S, D), all bfloat16, with the queries
    at the last L keys. rel: (B, L, Hq, R), the bias at distances below R.
    sinks: (B, L, Hq). key_start: (B,), the first key of each row. Returns
    (B, L, Hq, D)."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    return _relative_attention_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sinks.astype(mx.float32)),
            mx.contiguous(key_start.astype(mx.int32)),
            mx.array([L, S], dtype=mx.int32),
            mx.array(scale, dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("HQ", Hq),
            ("HKV", Hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", window),
            ("BQ", _BQ),
            ("TG", _TG),
        ],
        grid=((L + _BQ - 1) // _BQ * _TG, Hq, B),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )[0]
