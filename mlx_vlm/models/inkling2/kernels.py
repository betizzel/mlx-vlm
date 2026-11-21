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

