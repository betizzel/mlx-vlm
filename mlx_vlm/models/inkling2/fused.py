"""Inkling 2's one-token decode step, a layer visit in a few kernels.

Adapted from Locals (https://github.com/Generated-Computers/Locals, ``locals/inkling.py`` at 0931def8,
Copyright 2026 Arip Asadulaev, Apache License 2.0): the Metal kernels below are theirs, unchanged; the Python
side drives this package's Inkling 2 modules and caches instead of theirs.

A token passes 48 layer visits, and around each visit's weight reads (attention's projections, eight experts) the
model does a great many small things to vectors of 2304: norms, attention over depth, the loop's gates, four short
convolutions, clips, a bias by distance, a softcapped SwiGLU. As MLX operations that is about 125 a visit, 6000 a
token, each a kernel waiting for the one before (~13,500 graph nodes a token on an M5 Pro, 38 ms a token of which
the weight reads are about 20). Here each stretch between two weight reads is one kernel:

* ``mix``: a norm, attention over depth, the loop's gate and the layer norm; after attention also the convolution
  of its output and the residual. It makes the inputs of attention, of the experts and of the output layer, and
  the messages of the pass before.
* ``kv_prep``: from the projections to the clipped queries, the bias of every distance, and the new key and value
  (their convolutions, clip and norm).
* ``attend``: attention of the one query over a window with its bias, sink and output gate.
* ``route``: the router's scores and its choice of experts.
* ``combine``: the weighted sum of the experts, its convolution and the residual.

The weight reads stay MLX's (BF16 matrix-vector products, gather_qmm on NVFP4/MXFP8). A step is taken this way for
one token a row; prompts, verify blocks, padded rows and a global layer's attention past 1024 keys (this package's
split-K decode kernel or fused SDPA) run the reference, and both leave the same caches.
"""

from __future__ import annotations

import math
import os
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from ..cache import ArraysCache, BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache

_F32 = mx.float32
_KV = (KVCache, RotatingKVCache, BatchKVCache, BatchRotatingKVCache)
# MLX_VLM_INKLING_FUSED_DECODE=0 decodes with the reference modules instead (~20% slower). The two paths agree to
# BF16 rounding, but near-ties can tip a long reasoning answer either way: on long-document retrieval questions
# (greedy, effort none) the fused step looped in 5 of 19 documents and the reference in 2.
ENABLED = os.environ.get("MLX_VLM_INKLING_FUSED_DECODE", "1") != "0"

# Sums over a threadgroup: each lane's value summed over its simdgroup, the simdgroups' sums added by every thread.
_HEADER = """
#define REDUCE(n, vals, tots) \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) { const float s_ = simd_sum((vals)[i_]); if (lane == 0) buf[i_][sg] = s_; } \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) { float t_ = 0.0f; for (int g_ = 0; g_ < NS; g_++) t_ += buf[i_][g_]; (tots)[i_] = t_; }
#define REDUCE2(n, vals, tots) \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) { const float s_ = simd_sum((vals)[i_]); if (lane == 0) buf[i_][sg] = s_; } \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    if (sg == 0) for (int i_ = 0; i_ < (n); i_++) { \\
        const float s_ = simd_sum(lane < NS ? buf[i_][lane] : 0.0f); if (lane == 0) red[i_] = s_; } \\
    threadgroup_barrier(mem_flags::mem_threadgroup); \\
    for (int i_ = 0; i_ < (n); i_++) (tots)[i_] = red[i_];
inline float sigmoid_(float x) { return 1.0f / (1.0f + metal::precise::exp(-x)); }
"""

# A GPU thread takes 25 to 45 ns an iteration of a loop, and a step is one chain of kernels, each waiting for the
# one before: a kernel whose threads loop 2000 times costs 70 us however little it computes. So the work is spread
# over a threadgroup of 1024 (the most there is), a few hundred iterations a thread at most.

_MIX_SRC = """
    constexpr int NS = TG / 32;
    constexpr int DPT = (D + TG - 1) / TG;
    constexpr int NSRC = K + 1;
    constexpr int NCH = TG / RANK;  // (the gate: a thread a rank and a chunk of the dimensions)
    constexpr int CD = (D + NCH - 1) / NCH;
    const uint j = threadgroup_position_in_grid.x;  // an output a threadgroup: the loop's three gates side by side
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int M = x_shape[0];
    const float eps = prm[0];
    const int jq = Q == 1 ? 0 : j;
    threadgroup float buf[NRED][NS];
    threadgroup float red[NRED];

    // the state this reads: x, or after attention the convolution of x plus the residual
    float v[DPT];
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        float val = 0.0f;
        if (d < D) {
            val = float(x[size_t(row) * D + d]);
            if (CONV) {
                const size_t so = size_t(row) * 3 * D + d, to = size_t(d) * 4;
                float s0, s1, s2;
                if (SEQ) {  // rows are one text's tokens: a token's earlier inputs are the rows before it
                    const int r = int(row);
                    s0 = r >= 3 ? float(x[size_t(r - 3) * D + d]) : state[size_t(r) * D + d];
                    s1 = r >= 2 ? float(x[size_t(r - 2) * D + d]) : state[size_t(r + 1) * D + d];
                    s2 = r >= 1 ? float(x[size_t(r - 1) * D + d]) : state[size_t(r + 2) * D + d];
                } else {
                    s0 = state[so]; s1 = state[so + D]; s2 = state[so + 2 * D];
                }
                float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * val + val) * 0.4472135955f;
                h = float(T(h));
                new_state[so] = s1; new_state[so + D] = s2; new_state[so + 2 * D] = val;
                val = ADD ? float(res[size_t(row) * D + d]) + h : h;
                resid[size_t(row) * D + d] = val;
            }
        }
        v[c] = val;
    }
    float one[1], tot1[1];
    one[0] = 0.0f;
    for (int c = 0; c < DPT; c++) one[0] += v[c] * v[c];
    REDUCE2(1, one, tot1)
    const float inv = metal::precise::rsqrt(tot1[0] / D + 1e-6f);
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        v[c] *= inv;
        if (d < D && j == 0) partial[size_t(row) * D + d] = v[c];
    }

    // attention over depth: the block states and this state, by this output's query
    float acc[NSRC], w[NSRC];
    for (int i = 0; i < NSRC; i++) acc[i] = 0.0f;
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        if (d >= D) continue;
        const float qv = queries[size_t(jq) * D + d];
        for (int i = 0; i < NSRC; i++) acc[i] += (i < K ? blocks[(size_t(i) * M + row) * D + d] : v[c]) * qv;
    }
    REDUCE2(NSRC, acc, w)
    const float sc = metal::precise::rsqrt(float(D));
    float mw = -INFINITY, zw = 0.0f;
    for (int i = 0; i < NSRC; i++) mw = max(mw, w[i] * sc);
    for (int i = 0; i < NSRC; i++) { w[i] = metal::precise::exp(w[i] * sc - mw); zw += w[i]; }
    float o[DPT];
    for (int c = 0; c < DPT; c++) {
        const int d = tid + c * TG;
        float s = 0.0f;
        if (d < D) for (int i = 0; i < NSRC; i++) s += w[i] / zw * (i < K ? blocks[(size_t(i) * M + row) * D + d] : v[c]);
        o[c] = s;
    }

    // the loop's gate: some of the state forgotten, some of the pass before's message added
    if (GATE) {
        threadgroup float on[D];
        threadgroup float mn[D];
        threadgroup float pg[TG];
        threadgroup float low[RANK];
        float mm[DPT], two[2], tt[2];
        two[0] = 0.0f; two[1] = 0.0f;
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            mm[c] = d < D ? messages[(size_t(row) * 4 + MOFF + j) * D + d] : 0.0f;
            two[0] += o[c] * o[c]; two[1] += mm[c] * mm[c];
        }
        REDUCE2(2, two, tt)
        const float io = metal::precise::rsqrt(tt[0] / D + 1e-6f), im = metal::precise::rsqrt(tt[1] / D + 1e-6f);
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d < D) { on[d] = float(GT(o[c] * io)); mn[d] = float(GT(mm[c] * im)); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int r = tid % RANK, chunk = tid / RANK;  // the low-rank projection of both: a rank's row, a chunk a thread
        const size_t base = (size_t(j) * RANK + r) * 2 * D;
        float gp = 0.0f;
        for (int d = chunk * CD; d < min(D, (chunk + 1) * CD); d++)
            gp += float(down[base + d]) * on[d] + float(down[base + D + d]) * mn[d];
        pg[tid] = gp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (chunk == 0) {
            for (int c = 1; c < NCH; c++) gp += pg[c * RANK + r];
            low[r] = float(GT(gp));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d >= D) continue;
            const size_t fb = (size_t(j) * D + d) * RANK;
            float f = 0.0f, gn = 0.0f;
            for (int k = 0; k < RANK; k++) { f += forget[fb + k] * low[k]; gn += gain[fb + k] * low[k]; }
            o[c] = (1.0f - metal::precise::tanh(f)) * o[c] + metal::precise::tanh(gn) * mm[c];
        }
    }

    if (NORM) {
        float n2[1], nt[1];
        n2[0] = 0.0f;
        for (int c = 0; c < DPT; c++) n2[0] += o[c] * o[c];
        REDUCE2(1, n2, nt)
        const float invj = metal::precise::rsqrt(nt[0] / D + eps);
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d < D) out[(size_t(row) * QO + j) * D + d] = OT(o[c] * invj * float(norm_w[size_t(j % QN) * D + d]));
        }
    } else {
        for (int c = 0; c < DPT; c++) {
            const int d = tid + c * TG;
            if (d < D) out[(size_t(row) * QO + j) * D + d] = OT(o[c]);
        }
    }
"""

_KV_SRC = """
    constexpr int NS = TG / 32;
    constexpr int C = HKV * HD;
    const uint h = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const float qk_thr = prm[0], rel_thr = prm[1], eps = prm[2];
    const float tau_ = tau[row];
    threadgroup float buf[9][NS];
    threadgroup float rr[DR];

    // the head's sink logit, and in a global layer its key and value taps: small projections of the inputs
    float nine[9], dots[9];
    for (int i = 0; i < 9; i++) nine[i] = 0.0f;
    for (int d = t; d < D; d += TG) {
        nine[0] += float(xq[size_t(row) * D + d]) * float(sink_w[size_t(h) * D + d]);
        if (DYN && h < HKV) {
            const float kx = float(xk[size_t(row) * D + d]), vx = float(xv[size_t(row) * D + d]);
            for (int lag = 0; lag < 4; lag++) {
                nine[1 + lag] += kx * float(kdyn_w[size_t(h * 4 + lag) * D + d]);
                nine[5 + lag] += vx * float(vdyn_w[size_t(h * 4 + lag) * D + d]);
            }
        }
    }
    REDUCE(DYN ? 9 : 1, nine, dots)
    if (t == 0) sinks[size_t(row) * HQ + h] = T(dots[0]);

    float four[4], tot[4];
    const float qy = t < HD ? float(q[size_t(row) * QW + QOFF + h * HD + t]) * float(qscale[h * HD + t]) : 0.0f;
    const float ry = t < DR ? float(r[size_t(row) * RW + ROFF + h * DR + t]) * float(rscale[h * DR + t]) : 0.0f;
    float ky = 0.0f, vy = 0.0f;
    if (h < HKV && t < HD) {  // this key/value head's channel: the short convolutions over the last 4 tokens
        const int c = h * HD + t;
        const size_t so = size_t(row) * 3 * C + c;
        const float kx = float(k[size_t(row) * KW + KOFF + c]), vx = float(v[size_t(row) * VW + VOFF + c]);
        float k0, k1, k2, v0, v1, v2;
        if (SEQ) {  // rows are one text's tokens: a token's earlier inputs are the rows before it
            const int r = int(row);
            k0 = r >= 3 ? float(k[size_t(r - 3) * KW + KOFF + c]) : kstate[size_t(r) * C + c];
            k1 = r >= 2 ? float(k[size_t(r - 2) * KW + KOFF + c]) : kstate[size_t(r + 1) * C + c];
            k2 = r >= 1 ? float(k[size_t(r - 1) * KW + KOFF + c]) : kstate[size_t(r + 2) * C + c];
            v0 = r >= 3 ? float(v[size_t(r - 3) * VW + VOFF + c]) : vstate[size_t(r) * C + c];
            v1 = r >= 2 ? float(v[size_t(r - 2) * VW + VOFF + c]) : vstate[size_t(r + 1) * C + c];
            v2 = r >= 1 ? float(v[size_t(r - 1) * VW + VOFF + c]) : vstate[size_t(r + 2) * C + c];
        } else {
            k0 = kstate[so]; k1 = kstate[so + C]; k2 = kstate[so + 2 * C];
            v0 = vstate[so]; v1 = vstate[so + C]; v2 = vstate[so + 2 * C];
        }
        float kc, vc;
        if (DYN) {  // taps by lag (0: the token), learned ones plus this token's own, per head
            kc = ((float(ktaps[c]) + float(T(dots[1]))) * kx + (float(ktaps[C + c]) + float(T(dots[2]))) * k2
                + (float(ktaps[2 * C + c]) + float(T(dots[3]))) * k1 + (float(ktaps[3 * C + c]) + float(T(dots[4]))) * k0
                + kx) * 0.3333333333f;
            vc = ((float(vtaps[c]) + float(T(dots[5]))) * vx + (float(vtaps[C + c]) + float(T(dots[6]))) * v2
                + (float(vtaps[2 * C + c]) + float(T(dots[7]))) * v1 + (float(vtaps[3 * C + c]) + float(T(dots[8]))) * v0
                + vx) * 0.3333333333f;
        } else {
            const size_t to = size_t(c) * 4;
            kc = (float(ktaps[to]) * k0 + float(ktaps[to + 1]) * k1 + float(ktaps[to + 2]) * k2 + float(ktaps[to + 3]) * kx + kx)
                * 0.4472135955f;
            vc = (float(vtaps[to]) * v0 + float(vtaps[to + 1]) * v1 + float(vtaps[to + 2]) * v2 + float(vtaps[to + 3]) * vx + vx)
                * 0.4472135955f;
        }
        ky = float(T(kc)); vy = float(T(vc));
        kstate_new[so] = k1; kstate_new[so + C] = k2; kstate_new[so + 2 * C] = kx;
        vstate_new[so] = v1; vstate_new[so + C] = v2; vstate_new[so + 2 * C] = vx;
    }
    four[0] = qy * qy; four[1] = ry * ry; four[2] = ky * ky; four[3] = vy * vy;
    REDUCE(4, four, tot)

    if (t < HD) {
        float qv = float(T(qy * min(qk_thr / (metal::precise::sqrt(tot[0]) + 1e-6f), 1.0f)));
        if (TAU) qv = float(T(qv * tau_));
        qo[(size_t(row) * HQ + h) * HD + t] = T(qv);
        if (h < HKV) {
            const size_t xo = (size_t(row) * HKV + h) * HD + t;
            ko[xo] = T(ky * min(qk_thr / (metal::precise::sqrt(tot[2]) + 1e-6f), 1.0f));
            vo[xo] = T(vy * metal::precise::rsqrt(tot[3] / HD + eps));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < DR) {
        float rv = float(T(ry * min(rel_thr / (metal::precise::sqrt(tot[1]) + 1e-6f), 1.0f)));
        if (TAU) rv = float(T(rv * tau_));
        rr[t] = rv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int d = t; d < R; d += TG) {  // the bias of each distance
        float b = 0.0f;
        for (int i = 0; i < DR; i++) b += rr[i] * float(proj[size_t(i) * R + d]);
        bias[(size_t(row) * HQ + h) * R + d] = T(b);
    }
"""

_ATTEND_SRC = """
    constexpr int NS = TG / 32;
    constexpr int G = HQ / HKV;  // query heads of one key/value head: a threadgroup, the keys and values read once
    constexpr int CH = TG / HD;  // the chunks of keys a value's sum is made in
    constexpr int CK = (TG + CH - 1) / CH;
    const uint kvh = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int S = keys_shape[2], M = q_shape[0];
    const int brow = SEQ ? 0 : int(row);  // (rows of one text share its keys)
    const int idx = meta[0], count = counts[brow];
    const float scale = prm[0];
    threadgroup float p[G][TG];
    threadgroup float part[G][TG];
    threadgroup float buf[G][NS];
    threadgroup float red[G];
    const device T* kp = keys + size_t(brow) * keys_strides[0] + size_t(kvh) * keys_strides[1];
    const device T* vp = values + size_t(brow) * values_strides[0] + size_t(kvh) * values_strides[1];
    const size_t qb = (size_t(row) * HQ + kvh * G) * HD;  // the first of the G query heads

    // Keys by their place in time: the query's own is ``last`` (a text's tokens: the keys after it are later
    // tokens of the pass), a thread takes the key ``first + t``, in slot (place + idx) % S of a window written
    // round and round
    const int last = SEQ ? S - M + int(row) : S - 1;
    const int first = max(0, last - (TG - 1));
    const int place = first + int(t);
    const bool seen = place <= last && place >= S - count && (WINDOW == 0 || last - place < WINDOW);
    float sc[G], m[G], z[G];
    for (int g = 0; g < G; g++) sc[g] = -INFINITY;
    if (seen) {
        const device T* ks = kp + size_t((place + idx) % S) * keys_strides[2];
        for (int g = 0; g < G; g++) sc[g] = 0.0f;
        for (int d = 0; d < HD; d++) {
            const float kd = float(ks[d]);
            for (int g = 0; g < G; g++) sc[g] += float(q[qb + g * HD + d]) * kd;
        }
        const int dist = last - place;
        for (int g = 0; g < G; g++)
            sc[g] = sc[g] * scale + (dist < R ? float(bias[(size_t(row) * HQ + kvh * G + g) * R + dist]) : 0.0f);
    }
    for (int g = 0; g < G; g++) { const float v = simd_max(sc[g]); if (lane == 0) buf[g][sg] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (int g = 0; g < G; g++) {
        const float v = simd_max(lane < NS ? buf[g][lane] : -INFINITY);
        if (lane == 0) red[g] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < G; g++) {
        m[g] = max(red[g], float(sinks[size_t(row) * HQ + kvh * G + g]));
        const float e = seen ? metal::precise::exp(sc[g] - m[g]) : 0.0f;
        p[g][t] = e;
        z[g] = simd_sum(e);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < G; g++) if (lane == 0) buf[g][sg] = z[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (int g = 0; g < G; g++) {
        const float v = simd_sum(lane < NS ? buf[g][lane] : 0.0f);
        if (lane == 0) red[g] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // (the sink takes weight and has no value)
    for (int g = 0; g < G; g++) z[g] = red[g] + metal::precise::exp(float(sinks[size_t(row) * HQ + kvh * G + g]) - m[g]);

    const int d = t % HD, ch = t / HD;
    float o[G];
    for (int g = 0; g < G; g++) o[g] = 0.0f;
    for (int i = ch * CK; i < min(last - first + 1, (ch + 1) * CK); i++) {
        const float vd = float(vp[size_t((first + i + idx) % S) * values_strides[2] + d]);
        for (int g = 0; g < G; g++) o[g] += p[g][i] * vd;
    }
    for (int g = 0; g < G; g++) part[g][t] = o[g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ch == 0) for (int g = 0; g < G; g++) {
        float v = o[g];
        for (int c = 1; c < CH; c++) v += part[g][c * HD + d];
        v = float(T(v / z[g]));
        const size_t at = (kvh * G + g) * HD + d;
        out[size_t(row) * HQ * HD + at] = T(v * sigmoid_(float(gate[size_t(row) * GW + GOFF + at])));
    }
"""

_ROUTE_SRC = """
    constexpr int CHUNK = (D + PARTS - 1) / PARTS;
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const int e = tid / PARTS, part = tid % PARTS;
    threadgroup float partial[E * PARTS];
    threadgroup float score[E];
    threadgroup float chosen[K];
    float dot = 0.0f;  // an expert's score, PARTS threads to it
    for (int d = part * CHUNK; d < min(D, (part + 1) * CHUNK); d++) dot += float(x[size_t(row) * D + d]) * w[size_t(e) * D + d];
    partial[tid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {
        for (int i = 1; i < PARTS; i++) dot += partial[e * PARTS + i];
        score[e] = sigmoid_(dot);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {  // its place among score + bias (the lower expert first at a tie); the K first are chosen
        const float mine = score[e] + bias[e];
        int rank = 0;
        for (int i = 0; i < E; i++) {
            const float other = score[i] + bias[i];
            rank += (other > mine || (other == mine && i < e)) ? 1 : 0;
        }
        if (rank < K) { inds[size_t(row) * K + rank] = uint(e); chosen[rank] = score[e]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < K) {  // weights without the bias
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += chosen[k];
        weights[size_t(row) * K + tid] = chosen[tid] / (sum + 1e-20f) * prm[0];
    }
"""

_COMBINE_SRC = """
    const uint d = thread_position_in_grid.x;
    const uint row = thread_position_in_grid.y;
    float ysum[4];  // the experts' sum of this token, and (rows one text's tokens) of the three before it
    for (int j = 0; j < 4; j++) {
        const int r = int(row) - 3 + j;
        ysum[j] = 0.0f;
        if (r < 0 || (j < 3 && !SEQ)) continue;
        float y = 0.0f;
        for (int k = 0; k < K; k++) y += float(ys[(size_t(r) * K + k) * D + d]) * weights[size_t(r) * K + k];
        ysum[j] = float(T(y));
    }
    const float y = ysum[3];
    const size_t so = size_t(row) * 3 * D + d, to = size_t(d) * 4;
    float s0, s1, s2;
    if (SEQ) {
        const int r = int(row);
        s0 = r >= 3 ? ysum[0] : state[size_t(r) * D + d];
        s1 = r >= 2 ? ysum[1] : state[size_t(r + 1) * D + d];
        s2 = r >= 1 ? ysum[2] : state[size_t(r + 2) * D + d];
    } else {
        s0 = state[so]; s1 = state[so + D]; s2 = state[so + 2 * D];
    }
    float h = (taps[to] * s0 + taps[to + 1] * s1 + taps[to + 2] * s2 + taps[to + 3] * y + y) * 0.4472135955f;
    h = float(T(h));
    new_state[so] = s1; new_state[so + D] = s2; new_state[so + 2 * D] = y;
    out[size_t(row) * D + d] = T(resid[size_t(row) * D + d] + h);
"""

_kernels: dict = {}


def _kernel(name: str, inputs: List[str], outputs: List[str], source: str, contiguous: bool = True):
    if name not in _kernels:
        _kernels[name] = mx.fast.metal_kernel(name="locals_inkling_" + name, input_names=inputs, output_names=outputs,
                                              source=source, header=_HEADER, ensure_row_contiguous=contiguous)
    return _kernels[name]


_consts: dict = {}


def _f(*vals: float) -> mx.array:
    if vals not in _consts:
        _consts[vals] = mx.array(vals, dtype=_F32)
    return _consts[vals]


_SMALL: dict = {}


def _small(dtype, vals: tuple) -> mx.array:
    """A few numbers as an array, kept: a step makes the same ones again and again (a window's place, a position's
    scale), and each new array is a trip to the GPU."""
    key = (dtype, vals)
    out = _SMALL.get(key)
    if out is None:
        if len(_SMALL) > 8192:
            _SMALL.clear()
        out = _SMALL[key] = mx.array(vals, dtype=dtype)
    return out


_ZERO = {}


def _zero(dtype) -> mx.array:
    if dtype not in _ZERO:
        _ZERO[dtype] = mx.zeros((1,), dtype=dtype)
    return _ZERO[dtype]


def mix(x: mx.array, blocks: mx.array, queries: mx.array, *, norm_w: Optional[mx.array] = None, eps: float = 1e-6,
        conv=None, messages: Optional[mx.array] = None, gates=None, moff: int = 0, outputs: int = 0, shape=None,
        seq: bool = False):
    """The inputs a visit reads over depth. ``x`` (M, D): the state (after attention, with ``conv``, attention's
    output). ``blocks`` (K, M, D) float32: the block states. ``queries`` (Q, D). Returns (out, partial[, resid,
    new_state]): ``out`` (M, outputs or Q, D), each query's attention over the block states and the normalized
    state ``partial`` (M, D), then:

    * with ``messages`` (M, 4, D) and ``gates`` = (down (n, rank, 2D), forget (n, D, rank), gain (n, D, rank)), the
      loop's gate of output j with message ``moff + j`` (one query's output gated ``outputs`` ways);
    * with ``norm_w`` (1 or n, D), the layer norm of each, in x's dtype; float32 without.

    ``conv`` = (state (M, 3, D), taps (D, 4), residual (M, D) or None): the state is first the short convolution
    of x, plus the residual; ``resid`` (M, D) float32 is that sum and ``new_state`` the convolution's. ``shape``:
    another shape for ``out`` (a kernel's output costs nothing to shape; a reshape after it is one more node of a
    step's chain, 9 us each). ``seq``: the rows are one text's consecutive tokens, so the convolution of a row reads
    the rows before it and ``state`` (1, 3, D) only at the start; ``new_state`` (M, 3, D) is then the state after
    each token."""
    M, D = x.shape
    K, Q = blocks.shape[0], queries.shape[0]
    QO = outputs or Q
    gate = messages is not None
    rank = gates[0].shape[1] if gate else 1
    TG = 1024 if D >= 1024 else 256 if D >= 256 else 32
    assert TG % rank == 0, (TG, rank)
    state, taps, res = conv if conv is not None else (_zero(_F32), _zero(_F32), None)
    down, forget, gain = gates if gate else (_zero(x.dtype), _zero(_F32), _zero(_F32))
    T, OT = x.dtype, x.dtype if norm_w is not None else _F32
    out = _kernel("mix", ["x", "blocks", "queries", "state", "taps", "res", "messages", "down", "forget", "gain", "norm_w", "prm"],
                  ["out", "partial", "resid", "new_state"], _MIX_SRC)(
        inputs=[x, blocks, queries, state, taps, res if res is not None else _zero(T), messages if gate else _zero(_F32),
                down, forget, gain, norm_w if norm_w is not None else _zero(T), _f(eps)],
        template=[("T", T), ("OT", OT), ("GT", down.dtype), ("D", D), ("K", K), ("Q", Q), ("QO", QO),
                  ("QN", norm_w.shape[0] if norm_w is not None else 1), ("TG", TG), ("CONV", conv is not None),
                  ("ADD", res is not None), ("GATE", gate), ("NORM", norm_w is not None), ("MOFF", moff), ("RANK", rank),
                  ("NRED", max(K + 1, 2)), ("SEQ", seq)],
        grid=(QO * TG, M, 1), threadgroup=(TG, 1, 1),
        output_shapes=[shape or (M, QO, D), (M, D), (M, D) if conv is not None else (1,),
                       (M, 3, D) if conv is not None else (1,)],
        output_dtypes=[OT, _F32, _F32, _F32])
    return out if conv is not None else out[:2]


def kv_prep(q, r, k, v, at, xq, xk, xv, sink_w, kdyn_w, vdyn_w, kstate, vstate, ktaps, vtaps, qscale, rscale, proj, tau, *,
            heads: int, kv_heads: int, hd: int, qk_threshold: float, rel_threshold: float, eps: float, seq: bool = False):
    """From a token's projections to what attention takes. ``q``, ``r``, ``k``, ``v``: (M, ...) arrays holding the
    query, relative, key and value projections at the offsets ``at`` (four ints; one array may hold several:
    projections of one input are one matrix). ``xq``, ``xk``, ``xv`` (M, D): the inputs, for the sink logit of each
    head (``sink_w`` (heads, D)) and, in a global layer, the dynamic taps of the key and value convolutions
    (``kdyn_w``, ``vdyn_w`` (kv_heads * 4, D), over learned taps (4, C); else None and plain taps (C, 4)).
    Returns the clipped, scaled queries (M, heads, hd), the bias of every distance (M, heads, R), the new key and
    value (M, kv_heads, 1, hd) after their convolutions, clip and norm, the sink logits (M, heads) and the
    convolutions' new states. ``tau`` (M,) scales queries and bias, or None. ``seq``: the rows are one text's
    consecutive tokens (see mix)."""
    M, D = xq.shape
    dr, R = rscale.shape[-1], proj.shape[-1]
    T = xq.dtype
    C = kv_heads * hd
    TG = max(32, -(-hd // 32) * 32)
    dyn = kdyn_w is not None
    z = _zero(T)
    return _kernel("kv", ["q", "r", "k", "v", "xq", "xk", "xv", "sink_w", "kdyn_w", "vdyn_w", "kstate", "vstate", "ktaps", "vtaps",
                          "qscale", "rscale", "proj", "tau", "prm"],
                   ["qo", "bias", "ko", "vo", "sinks", "kstate_new", "vstate_new"], _KV_SRC)(
        inputs=[q, r, k, v, xq, xk if dyn else z, xv if dyn else z, sink_w, kdyn_w if dyn else z, vdyn_w if dyn else z, kstate,
                vstate, ktaps, vtaps, qscale, rscale, proj, tau if tau is not None else _zero(_F32),
                _f(qk_threshold, rel_threshold, eps)],
        template=[("T", T), ("D", D), ("HQ", heads), ("HKV", kv_heads), ("HD", hd), ("DR", dr), ("R", R), ("TG", TG), ("DYN", dyn),
                  ("TAU", tau is not None), ("QW", q.shape[-1]), ("RW", r.shape[-1]), ("KW", k.shape[-1]), ("VW", v.shape[-1]),
                  ("QOFF", at[0]), ("ROFF", at[1]), ("KOFF", at[2]), ("VOFF", at[3]), ("SEQ", seq)],
        grid=(heads * TG, M, 1), threadgroup=(TG, 1, 1),
        output_shapes=[(M, heads, hd), (M, heads, R), (M, kv_heads, 1, hd), (M, kv_heads, 1, hd), (M, heads), (M, 3, C), (M, 3, C)],
        output_dtypes=[T, T, T, T, T, _F32, _F32])


def window(hd: int) -> int:
    """The most keys attend() takes for heads of ``hd``: a thread a key, in one threadgroup."""
    return min(1024, -(-hd * 8 // 32) * 32)


def attend(q, keys, values, bias, sinks, gate, idx: int, counts, scale: float, gate_at: int = 0, window_size: int = 0,
           seq: bool = False):
    """Attention of each row's one query (M, heads, hd) over its window, ``keys`` and ``values`` (M, kv_heads, S,
    hd) with S <= window(hd), as the cache holds them: ``idx`` is the slot after the newest key (0: the keys are in order) and
    ``counts`` (M,) int32 each row's number of real keys, its newest (a batch's shorter rows are padded). With the bias by distance (M, heads, R), the sink logits (M, heads) and the sigmoid of
    ``gate`` on the output (M, heads * hd, or wider with the gate at ``gate_at``): (M, heads * hd). ``seq``: the
    M queries are one text's consecutive tokens over its keys (1, kv_heads, S, hd), which end with theirs, in
    order (``idx`` 0): each sees the keys up to its own, the last ``window_size`` of them in a sliding layer."""
    M, heads, hd = q.shape
    T = q.dtype
    TG = window(hd)
    return _kernel("attend", ["q", "keys", "values", "bias", "sinks", "gate", "meta", "counts", "prm"], ["out"], _ATTEND_SRC, False)(
        inputs=[q, keys, values, bias, sinks, gate, _small(mx.int32, (idx,)), counts, _f(scale)],
        template=[("T", T), ("HQ", heads), ("HKV", keys.shape[1]), ("HD", hd), ("R", bias.shape[-1]), ("TG", TG), ("GW", gate.shape[-1]),
                  ("GOFF", gate_at), ("WINDOW", window_size), ("SEQ", seq)],
        grid=(keys.shape[1] * TG, M, 1), threadgroup=(TG, 1, 1), output_shapes=[(M, heads * hd)], output_dtypes=[T])[0]


def route(x, weight, bias, k: int, route_scale: float):
    """The router: of each row (M, D) the ``k`` experts with the highest sigmoid score plus bias, uint32, and
    their weights (the scores, normalized to sum to ``route_scale``)."""
    E, D = weight.shape
    M = x.size // D
    parts = max(1, min(8, 1024 // E))
    return _kernel("route", ["x", "w", "bias", "prm"], ["inds", "weights"], _ROUTE_SRC)(
        inputs=[x, weight, bias, _f(route_scale)], template=[("T", x.dtype), ("D", D), ("E", E), ("K", k), ("PARTS", parts)],
        grid=(M * E * parts, 1, 1), threadgroup=(E * parts, 1, 1), output_shapes=[(M, k), (M, k)],
        output_dtypes=[mx.uint32, _F32])


def combine(ys, weights, state, taps, resid, seq: bool = False):
    """The experts' outputs (M, k, ..., D) summed by their weights, the short convolution of that (state (M, 3, D), taps
    (D, 4)) and the residual (M, D) float32: the visit's output (M, D) and the convolution's new state."""
    M, D = resid.shape
    K = ys.size // (M * D)
    T = ys.dtype
    return _kernel("combine", ["ys", "weights", "state", "taps", "resid"], ["out", "new_state"], _COMBINE_SRC)(
        inputs=[ys, weights, state, taps, resid], template=[("T", T), ("K", K), ("D", D), ("SEQ", seq)], grid=(D, M, 1),
        threadgroup=(min(256, D), 1, 1), output_shapes=[(M, D), (M, 3, D)], output_dtypes=[T, _F32])


def _rms(x: mx.array) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + 1e-6)


class _Consts:
    """A layer's arrays in the layouts the kernels read, made once. The projections that read one input (queries,
    relative, output gate, then keys and values) become row slices of one BF16 matrix: one read where there were
    five, and nothing held twice (the modules keep their rows as views of it)."""

    def __init__(self, layer, cfg):
        A, D = layer.self_attn, cfg.hidden_size
        f32 = lambda a: a.astype(_F32)
        self.attn_query = (
            f32(layer.attention_input.query).reshape(-1, D) if layer.attention_input is not None else None
        )
        norms = (
            (layer.input_layernorm,)
            if layer.is_sliding
            else (layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm)
        )
        self.attn_norm = mx.stack([n.weight for n in norms])
        self.attn_eps = norms[0].eps
        self.mlp_query = f32(layer.mlp_input.query).reshape(1, D)
        self.mlp_norm = layer.post_attention_layernorm.weight.reshape(1, D)
        self.mlp_eps = layer.post_attention_layernorm.eps
        self.attn_taps = f32(layer.attn_sconv.conv.weight).reshape(D, -1)
        self.mlp_taps = f32(layer.mlp_sconv.conv.weight).reshape(D, -1)
        if layer.is_sliding:
            self.k_taps = f32(A.k_sconv.conv.weight).reshape(-1, A.k_sconv.kernel_size)
            self.v_taps = f32(A.v_sconv.conv.weight).reshape(-1, A.v_sconv.kernel_size)
            self.kdyn_w = self.vdyn_w = None
        else:
            self.k_taps, self.v_taps = A.k_sconv.static_weight, A.v_sconv.static_weight
            self.kdyn_w, self.vdyn_w = A.k_sconv.weight_proj.weight, A.v_sconv.weight_proj.weight
        self.attn_gates = self.mlp_gates = None
        if hasattr(layer, "q_message_gate"):

            def stack(gates):
                return (
                    mx.stack([g.down_proj.weight for g in gates]),
                    mx.stack([f32(g.forget_gate_proj.weight) for g in gates]),
                    mx.stack([f32(g.message_gate_proj.weight) for g in gates]),
                )

            self.attn_gates = stack((layer.q_message_gate, layer.k_message_gate, layer.v_message_gate))
            self.mlp_gates = stack((layer.mlp_message_gate,))
        self.router_w = f32(layer.mlp.gate_weight)
        self.router_b = f32(layer.mlp.e_score_correction_bias)
        self.sink_w = A.sink_proj.weight
        mods = [A.q_proj, A.r_proj, A.gate_proj, A.k_proj, A.v_proj]
        # BF16, or quantized (any mode quantized_matmul takes): row slices of the same arrays either way, since
        # quantization groups run along the input dimension.
        names = [n for n in ("weight", "scales", "biases") if n in mods[0]]
        parts = [mx.concatenate([m[n] for m in mods]) for n in names]
        mx.eval(parts)
        self.at, start = [], 0
        for m in mods:
            rows = m.weight.shape[0]
            self.at.append(start)
            for n, p in zip(names, parts):
                setattr(m, n, p[start : start + rows])
            start += rows
        q = mods[0]
        self.quant = (q.group_size, q.bits, q.mode) if isinstance(q, nn.QuantizedLinear) else None
        self.stack = parts
        self.head = [p[: self.at[3]] for p in parts]
        mx.eval([v for v in vars(self).values() if isinstance(v, mx.array)], [m.parameters() for m in mods])


def _project(x: mx.array, parts, quant) -> mx.array:
    """``x`` times the stacked projections ``parts`` (weight[, scales[, biases]]), transposed."""
    if quant is None:
        return x @ parts[0].T
    group_size, bits, mode = quant
    return mx.quantized_matmul(
        x,
        parts[0],
        parts[1],
        parts[2] if len(parts) > 2 else None,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def _plain_projections(text_model) -> bool:
    """Unsharded attention projections without biases that _Consts can stack into one matrix: in each layer all BF16
    / FP32 ``nn.Linear``, or all ``nn.QuantizedLinear`` of one group size, width and mode."""
    ok = getattr(text_model, "_fused_plain", None)
    if ok is None:

        def stackable(A):
            mods = (A.q_proj, A.r_proj, A.gate_proj, A.k_proj, A.v_proj)
            if A.sharding_group is not None or any("bias" in m for m in mods):
                return False
            if all(type(m) is nn.Linear for m in mods):
                return True
            first = mods[0]
            return all(
                type(m) is nn.QuantizedLinear
                and (m.group_size, m.bits, m.mode) == (first.group_size, first.bits, first.mode)
                for m in mods
            )

        ok = all(stackable(layer.self_attn) for layer in text_model.layers)
        object.__setattr__(text_model, "_fused_plain", ok)
    return ok


def supported(text_model, h: mx.array, cache) -> bool:
    """Whether this step can go in kernels: one token of one row (the server decodes one sequence; the long-context
    attention below takes one row), on Metal, BF16 or FP32, every visit's caches of the kinds the kernels read,
    nothing speculative and nothing padded."""
    B, L = h.shape[:2]
    if not ENABLED or B != 1 or L != 1 or cache is None or not mx.metal.is_available() or mx.default_device() != mx.gpu:
        return False
    if h.dtype not in (mx.bfloat16, mx.float32) or not _plain_projections(text_model):
        return False
    for entry in cache:
        kv, conv = entry[0], entry[1]
        if type(kv) not in _KV or not isinstance(conv, ArraysCache):
            return False
        if getattr(kv, "is_speculating", False) or getattr(conv, "is_speculating", False):
            return False
        if getattr(conv, "_lengths", None) is not None or not _unpadded(conv):
            return False
    return True


def _unpadded(conv) -> bool:
    """Whether a batch row's left padding is all behind it. Padding only shrinks as the cache advances, so its
    largest value is read from the GPU once per padding array (the server sets a new one per request), not on
    every step: reading it waits for the step in flight and stalls the pipeline."""
    base = getattr(conv, "_left_padding", None)
    if base is None:
        return True
    seen = getattr(conv, "_fused_padding", None)
    if seen is None or seen[0] is not base:
        seen = (base, int(mx.max(base)))
        conv._fused_padding = seen
    return conv._left_padding_advance >= seen[1]


def decode_step(text_model, h: mx.array, cache) -> mx.array:
    """One token a row through every layer visit in kernels. ``h`` (B, 1, D): the embedded tokens. Returns the
    pre-norm output (B, 1, D) float32, as the reference's ``skip_final_norm`` does."""
    cfg = text_model.config
    consts = getattr(text_model, "_fused_consts", None)
    if consts is None:
        consts = [_Consts(layer, cfg) for layer in text_model.layers]
        object.__setattr__(text_model, "_fused_consts", consts)
    x = h[:, 0]
    start = cfg.loop_layer_index_start
    blocks = pre_loop = bank = None
    for i, cache_idx, p in text_model.schedule():
        if i == start:
            if p == 0:
                pre_loop = (x, blocks)
            else:
                done = 0 if pre_loop[1] is None else pre_loop[1].shape[0]
                bank = (blocks[done + 1 :], x)
                x, blocks = pre_loop
        messages = None
        if p > 0:
            messages = mix(bank[1], bank[0], text_model.message_input.query[i - start].astype(_F32))[0]
        entry = cache[cache_idx]
        x, blocks = _visit(text_model.layers[i], consts[i], cfg, x, blocks, entry[0], entry[1], messages)
    out = mix(x, blocks, text_model.final_input.query.reshape(1, -1).astype(_F32))[0]
    return out.reshape(h.shape[0], 1, -1)


def _long_attention(A, qo, keys, values, bias, sinks, kv):
    """A global layer's single query past the attend kernel's window: the split-K kernel from 4,096 keys, fused
    SDPA with the bias as a mask below that, as Attention._decode_attention does."""
    from .kernels import decode_attention
    from .language import _DECODE_KERNEL_MIN_KEYS

    q = qo[:, :, None, :]
    size = keys.shape[2]
    start = kv.left_padding[:1] if isinstance(kv, BatchKVCache) else None
    if qo.shape[0] == 1 and size >= _DECODE_KERNEL_MIN_KEYS and q.dtype == mx.bfloat16 and A.head_dim % 32 == 0:
        return decode_attention(
            q,
            keys,
            values,
            bias[0],
            sinks[0],
            scale=A.head_dim**-0.5,
            key_start=None if start is None else mx.maximum(start, 0),
        )
    distance = (size - 1) - mx.arange(size)
    rel = mx.take(bias, mx.minimum(distance, A.rel_extent - 1), axis=-1)
    rel = mx.where(distance < A.rel_extent, rel, mx.array(0, rel.dtype))
    if start is not None:
        rel = mx.where(mx.arange(size) >= start[:, None, None], rel, mx.array(-mx.inf, rel.dtype))
    return mx.fast.scaled_dot_product_attention(
        q, keys, values, scale=A.head_dim**-0.5, mask=rel[:, :, None, :], sinks=sinks[0].astype(q.dtype)
    )


def _visit(layer, c: _Consts, cfg, x, blocks, kv, conv, messages):
    """One visit of the rows ``x`` (M, D), a token of each text: the visit's output (M, D) and the block states."""
    A = layer.self_attn
    M, D = x.shape
    C = A.k_proj.weight.shape[0]
    st = [
        mx.zeros((M, 3, n), _F32) if s is None else s.astype(_F32)
        for s, n in zip(conv.cache, (C, C, D, D))
    ]
    if layer.attention_input is None:  # the first layer reads the embeddings as they are
        normed = _rms(x.astype(_F32))
        xq = xk = xv = layer.input_layernorm(x)
    else:
        gated = messages is not None
        out, normed = mix(
            x,
            blocks,
            c.attn_query,
            norm_w=c.attn_norm,
            eps=c.attn_eps,
            messages=messages,
            gates=c.attn_gates if gated else None,
            outputs=3 if gated else 0,
            shape=(M, D) if layer.is_sliding and not gated else None,
        )
        xq, xk, xv = (out,) * 3 if out.ndim == 2 else (out[:, 0], out[:, 1], out[:, 2])
    tau = None
    batch = isinstance(kv.offset, mx.array)
    if not layer.is_sliding and cfg.log_scaling_n_floor is not None:  # queries sharpen with the log of the position
        if batch:
            pos = (kv.offset + 1).astype(_F32)
            tau = 1.0 + cfg.log_scaling_alpha * mx.log(mx.maximum(pos / cfg.log_scaling_n_floor, 1.0))
        else:
            p = kv.offset + 1
            value = 1.0 + cfg.log_scaling_alpha * math.log(max(p / cfg.log_scaling_n_floor, 1.0))
            tau = _small(_F32, (value,) * M)
    if xk is xq and xv is xq:  # one input: every projection in one read
        q = r = k = v = gate = _project(xq, c.stack, c.quant)
        at, gate_at = (c.at[0], c.at[1], c.at[3], c.at[4]), c.at[2]
    else:
        q = r = gate = _project(xq, c.head, c.quant)
        k, v, at, gate_at = A.k_proj(xk), A.v_proj(xv), (c.at[0], c.at[1], 0, 0), c.at[2]
    qo, bias, ko, vo, sinks, ks, vs = kv_prep(
        q,
        r,
        k,
        v,
        at,
        xq,
        xk,
        xv,
        c.sink_w,
        c.kdyn_w,
        c.vdyn_w,
        st[0],
        st[1],
        c.k_taps,
        c.v_taps,
        A.q_scale,
        A.rel_scale,
        A.rel_proj,
        tau,
        heads=A.n_heads,
        kv_heads=A.n_kv,
        hd=A.head_dim,
        qk_threshold=cfg.qk_l2_clip_threshold,
        rel_threshold=cfg.rel_l2_clip_threshold,
        eps=cfg.rms_norm_eps,
    )
    keys, values = kv.update_and_fetch(ko, vo)
    S = keys.shape[2]
    if S <= window(qo.shape[-1]):
        idx = kv._idx if isinstance(kv, (RotatingKVCache, BatchRotatingKVCache)) else 0
        counts = mx.minimum(kv.offset, S).astype(mx.int32) if batch else _small(mx.int32, (S,) * M)
        a = attend(qo, keys, values, bias, sinks, gate, idx, counts, A.head_dim**-0.5, gate_at, A.sliding)
    else:  # a global layer past the kernel's window
        g = gate[:, gate_at : gate_at + qo.shape[1] * qo.shape[2]]
        a = _long_attention(A, qo, keys, values, bias, sinks, kv)
        a = (a.reshape(M, -1).astype(_F32) * mx.sigmoid(g.astype(_F32))).astype(g.dtype)
    a = A.o_proj(a)

    boundary = layer.is_block_boundary
    if boundary:
        blocks = normed[None] if blocks is None else mx.concatenate([blocks, normed[None]])
    gated = messages is not None
    mlp_in, _, resid, cs = mix(
        a,
        blocks,
        c.mlp_query,
        norm_w=c.mlp_norm,
        eps=c.mlp_eps,
        conv=(st[2], c.attn_taps, None if boundary else x),
        messages=messages,
        gates=c.mlp_gates if gated else None,
        moff=3,
        shape=(M, 1, D),
    )
    moe = layer.mlp
    inds, weights = route(mlp_in, c.router_w, c.router_b, moe.top_k, moe.route_scale)
    ys = moe.switch_mlp(mlp_in, inds.reshape(M, 1, -1))
    x, ms = combine(ys, weights, st[3], c.mlp_taps, resid)
    for i, s in enumerate((ks, vs, cs, ms)):
        conv[i] = s
    conv.advance(1)
    return x, blocks
