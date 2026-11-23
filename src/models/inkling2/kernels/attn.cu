// src/models/inkling2/kernels/attn.cu - Inkling 2's decode attention: a query head's scores take a per-query relative bias
// over the last kExtent positions, and a per-query sink logit sits in the softmax's denominator.
//
// A row's keys split in slices whose size comes from the row's own key count alone (whole tiles of the block's four warps,
// at most kMaxSplits slices), so a row's bits do not depend on the other rows of its call or on the launch's bound. A block
// takes one row, one key/value head (its three query heads) and one slice; each warp walks its tiles of 32 keys with a lane
// a key: the lane loads its key's row and scores it in registers, puts its value row in shared memory, and the lanes then
// sum the values with four channels a lane. A second kernel merges a row's slices with the sink, gates the heads and rounds
// them to the q8_1 blocks of the output projection.
#include "basalt/kernels/pdl.hpp"
#include "basalt/kernels/q8_1_finite.hpp"
#include "basalt/models/inkling2/kv_row_codec.hpp"
#include "basalt/models/inkling2/kernels.hpp"
#include "prep.cuh"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace basalt::models::inkling2 {
namespace {

namespace bk = basalt::kernels;

constexpr int kWarps = 4;
constexpr int kTile = 32;
constexpr int kUnit = kWarps * kTile;        // keys of one pass of the block's warps
constexpr int kGroup = kHeads / kKvHeads;    // 3 query heads a key/value head
constexpr int kMaxSplits = 64;
constexpr int kVStride = kHeadDim / 2 + 4;   // words a value row takes in shared memory (the lanes' stores do not collide)

struct Q81 {
    half2 ds;
    int8_t qs[32];
};

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling2 ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
    return v;
}

// The keys a slice of a row with n keys covers: whole passes of the block, at most kMaxSplits slices.
__host__ __device__ __forceinline__ int span_of(int n) {
    const int units = (n + kUnit * kMaxSplits - 1) / (kUnit * kMaxSplits);
    return (units > 1 ? units : 1) * kUnit;
}

__device__ __forceinline__ void row_keys(const AttnArgs& a, int r, int& p, int& lo, int& n) {
    p = a.pos[r];
    lo = a.window > 0 ? max(0, p - a.window + 1) : 0;
    n = p + 1 - lo;
}

// 8 cache values as fp32 from 16 bytes of fp16 (or the first 8 of 16 bytes of E4M3, `hi` picking the other 8)
__device__ __forceinline__ void unpack8_f16(const uint4 u, float (&f)[8]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 t = __half22float2(*reinterpret_cast<const __half2*>(&w[i]));
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}
__device__ __forceinline__ void unpack16_fp8(const uint4 u, float (&f)[16]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        __nv_fp8x2_e4m3 lo, hi;
        lo.__x = (__nv_fp8x2_storage_t) (w[i] & 0xFFFF);
        hi.__x = (__nv_fp8x2_storage_t) (w[i] >> 16);
        const float2 a = (float2) lo, b = (float2) hi;
        f[4 * i] = a.x;
        f[4 * i + 1] = a.y;
        f[4 * i + 2] = b.x;
        f[4 * i + 3] = b.y;
    }
}
// 16 INT8 codes times their fp16 scale as fp32 (exact: a code times an fp16 scale fits fp32's mantissa). A code's float is
// its byte, biased by 128, placed under 2^23's exponent, less 2^23 + 128: no integer conversion.
__device__ __forceinline__ void unpack16_int8(const uint4 u, float scale, float (&f)[16]) {
    const uint32_t w[4] = {u.x ^ 0x80808080u, u.y ^ 0x80808080u, u.z ^ 0x80808080u, u.w ^ 0x80808080u};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            f[4 * i + j] = (__uint_as_float(__byte_perm(w[i], 0x4b000000u, 0x7650u | j)) - 8388736.f) * scale;
}
// 16 INT8 codes times their fp16 scale, rounded to fp16 pairs: the same bits as the fp32 product rounded (the product of a
// code and an fp16 scale is exact before the one rounding). A code's half is its biased byte under 1024's exponent, less 1152.
__device__ __forceinline__ void unpack16_int8_h2(const uint4 u, __half2 scale, uint32_t (&h)[8]) {
    const uint32_t w[4] = {u.x ^ 0x80808080u, u.y ^ 0x80808080u, u.z ^ 0x80808080u, u.w ^ 0x80808080u};
    const __half2 bias = __half2half2(__ushort_as_half(0x6480u));   // 1152
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint32_t b = __byte_perm(w[i], 0x64646464u, j ? 0x7362u : 0x5140u);
            const __half2 v = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&b), bias), scale);
            h[2 * i + j] = *reinterpret_cast<const uint32_t*>(&v);
        }
}
// A row's two fp16 scales (after its kHeadDim codes) as fp32.
__device__ __forceinline__ void row_scales(const uint8_t* row, float (&s)[2]) {
    const __half2 h = *reinterpret_cast<const __half2*>(row + kHeadDim);
    s[0] = __low2float(h);
    s[1] = __high2float(h);
}

template <KVFormat F>
struct Smem {
    float q[kGroup][kHeadDim];
    float rs[kGroup][kRel];
    uint32_t v[kWarps][kTile][kVStride];   // a tile's value rows as fp16 pairs
    float m[kWarps][kGroup], l[kWarps][kGroup];
    float acc[kWarps][kGroup][kHeadDim];
    // the tile's rows' MXFP8 group scales (4 bytes a row); fp16/E4M3 rows are unit-scale and carry none
    uint8_t vsc[kWarps][kTile][F == KVFormat::Mxfp8 ? kMxfp8Groups : 1];
};

template <KVFormat F>
__global__ void __launch_bounds__(kWarps * 32) attn_split_kernel(AttnArgs a, int z, float* __restrict__ part_ml,
                                                                 float* __restrict__ part_acc) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<F>& sm = *reinterpret_cast<Smem<F>*>(smem_raw);
    const int r = blockIdx.x, g = blockIdx.y, s = blockIdx.z;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    int p, lo, n;
    row_keys(a, r, p, lo, n);
    const int span = span_of(n);
    if (s * span >= n) return;   // past this row's slices
    const int s_lo = lo + s * span, s_hi = min(p + 1, s_lo + span);
    constexpr int kRowBytes = F == KVFormat::Fp16 ? kHeadDim * 2 : F == KVFormat::Int8 ? kKvInt8RowBytes : kHeadDim;
    constexpr int kLoads = F == KVFormat::Fp16 ? kHeadDim / 8 : kHeadDim / 16;
    const SegCaches& sc = a.seg.c[bk::seg_of(a.seg.at, r)];   // a batch window's row: its conversation's caches
    const uint8_t* kbase = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.k : a.k_cache) + (size_t) g * kRowBytes;
    const uint8_t* vbase = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.v : a.v_cache) + (size_t) g * kRowBytes;
    const uint8_t* kscales = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.k : a.k_cache) + code_bytes(F) + 4 * g;
    const uint8_t* vscales = static_cast<const uint8_t*>(a.seg.at.n > 0 ? sc.v : a.v_cache) + code_bytes(F) + 4 * g;
    // a tile's key and value rows, the lane's key's (zero past the slice); the warp's first tile is loaded with the queries
    uint8_t ke[4] = {0, 0, 0, 0}, ve[4] = {0, 0, 0, 0};
    float ks[2] = {1.f, 1.f}, vs[2] = {1.f, 1.f};
    const auto load_tile = [&](int t0, uint4 (&kr)[kLoads], uint4 (&vr)[kLoads]) {
        const int key = t0 + lane;
        ks[0] = ks[1] = vs[0] = vs[1] = 1.f;
        ke[0] = ke[1] = ke[2] = ke[3] = ve[0] = ve[1] = ve[2] = ve[3] = 0;
        if (key < s_hi) {
            const size_t cell = (size_t) (key % a.cap) * cell_bytes(F);
            const uint4* kp = reinterpret_cast<const uint4*>(kbase + cell);
            const uint4* vp = reinterpret_cast<const uint4*>(vbase + cell);
#pragma unroll
            for (int i = 0; i < kLoads; ++i) kr[i] = kp[i];
#pragma unroll
            for (int i = 0; i < kLoads; ++i) vr[i] = vp[i];
            if constexpr (F == KVFormat::Int8) {
                row_scales(kbase + cell, ks);
                row_scales(vbase + cell, vs);
            } else if constexpr (F == KVFormat::Mxfp8) {
                const uint8_t* ksp = kscales + cell;
                const uint8_t* vsp = vscales + cell;
#pragma unroll
                for (int j = 0; j < 4; ++j) { ke[j] = ksp[j]; ve[j] = vsp[j]; }
            }
        } else {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) kr[i] = vr[i] = make_uint4(0, 0, 0, 0);
        }
    };
    const int t_first = s_lo + w * kTile;
    uint4 kr[kLoads], vr[kLoads];
    if (t_first < s_hi) load_tile(t_first, kr, vr);
    for (int i = threadIdx.x; i < kGroup * kHeadDim; i += blockDim.x)
        sm.q[i / kHeadDim][i % kHeadDim] = a.q[((size_t) r * kHeads + g * kGroup) * kHeadDim + i];
    for (int i = threadIdx.x; i < kGroup * kRel; i += blockDim.x)
        sm.rs[i / kRel][i % kRel] = a.r[((size_t) r * kHeads + g * kGroup) * kRel + i];
    __syncthreads();
    float m[kGroup], l[kGroup], acc[kGroup][4];
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.f;
#pragma unroll
        for (int u = 0; u < 4; ++u) acc[h][u] = 0.f;
    }
    for (int t0 = t_first; t0 < s_hi; t0 += kUnit) {
        const int key = t0 + lane;
        const bool valid = key < s_hi;
        if (t0 != t_first) load_tile(t0, kr, vr);
        // the lane's key scored for the three heads
        float sc[kGroup] = {0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            if constexpr (F == KVFormat::Fp16) {
                float f[8];
                unpack8_f16(kr[i], f);
#pragma unroll
                for (int d = 0; d < 8; ++d)
#pragma unroll
                    for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[h][8 * i + d], f[d], sc[h]);
            } else {
                float f[16];
                if constexpr (F == KVFormat::Fp8) unpack16_fp8(kr[i], f);
                else if constexpr (F == KVFormat::Int8) unpack16_int8(kr[i], ks[i / 4], f);
                else mxfp8_decode16(kr[i], mxfp8_scale_factor(ke[i >> 1]), f);
#pragma unroll
                for (int d = 0; d < 16; ++d)
#pragma unroll
                    for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[h][16 * i + d], f[d], sc[h]);
            }
        }
        const int dist = p - key;
        float tv[kRel];
#pragma unroll
        for (int i = 0; i < kRel; ++i) tv[i] = valid && dist < kExtent ? a.rel_table[i * kExtent + dist] : 0.f;
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            float b = 0.f;
#pragma unroll
            for (int i = 0; i < kRel; ++i) b = fmaf(sm.rs[h][i], tv[i], b);
            sc[h] = valid ? sc[h] * a.scale + b : -INFINITY;
        }
        // the lane's value row into shared memory as fp16 pairs (E4M3 is exact in fp16); the MXFP8 group scale is
        // applied in the accumulate step, so the shared row is the codes as for unit-scale E4M3
        uint32_t* vrow = sm.v[w][lane];
        if constexpr (F == KVFormat::Fp16) {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) *reinterpret_cast<uint4*>(vrow + 4 * i) = vr[i];
        } else if constexpr (F == KVFormat::Int8) {
#pragma unroll
            for (int i = 0; i < kLoads; ++i) {
                uint32_t h[8];
                unpack16_int8_h2(vr[i], __float2half2_rn(vs[i / 4]), h);
#pragma unroll
                for (int d = 0; d < 8; ++d) vrow[8 * i + d] = h[d];
            }
        } else {   // fp8 and mxfp8
#pragma unroll
            for (int i = 0; i < kLoads; ++i) {
                float f[16];
                unpack16_fp8(vr[i], f);
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    const __half2 hv = __floats2half2_rn(f[2 * d], f[2 * d + 1]);
                    vrow[8 * i + d] = *reinterpret_cast<const uint32_t*>(&hv);
                }
            }
            if constexpr (F == KVFormat::Mxfp8) {   // the row's group scales: the accumulate applies them (the shared row is the codes)
#pragma unroll
                for (int j = 0; j < 4; ++j) sm.vsc[w][lane][j] = ve[j];
            }
        }
        // the online softmax over the tile
        float pr[kGroup];
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const float mt = warp_max(sc[h]);
            const float mn = fmaxf(m[h], mt);
            const float corr = m[h] == -INFINITY ? 0.f : expf(m[h] - mn);
            pr[h] = sc[h] == -INFINITY ? 0.f : expf(sc[h] - mn);
            l[h] = l[h] * corr + warp_sum(pr[h]);
#pragma unroll
            for (int u = 0; u < 4; ++u) acc[h][u] *= corr;
            m[h] = mn;
        }
        __syncwarp();
        for (int j = 0; j < kTile; ++j) {
            const uint2 u = *reinterpret_cast<const uint2*>(&sm.v[w][j][lane * 2]);
            float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
            float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
            if constexpr (F == KVFormat::Mxfp8) {
                const float scale = mxfp8_scale_factor(sm.vsc[w][j][lane >> 3]);
                v01.x *= scale; v01.y *= scale; v23.x *= scale; v23.y *= scale;
            }
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                const float pj = __shfl_sync(0xffffffffu, pr[h], j);
                acc[h][0] = fmaf(pj, v01.x, acc[h][0]);
                acc[h][1] = fmaf(pj, v01.y, acc[h][1]);
                acc[h][2] = fmaf(pj, v23.x, acc[h][2]);
                acc[h][3] = fmaf(pj, v23.y, acc[h][3]);
            }
        }
        __syncwarp();
    }
    // the four warps' parts of this slice, merged in warp order
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        if (lane == 0) {
            sm.m[w][h] = m[h];
            sm.l[w][h] = l[h];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) sm.acc[w][h][lane * 4 + u] = acc[h][u];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kGroup * kHeadDim; i += blockDim.x) {
        const int h = i / kHeadDim, d = i % kHeadDim;
        float mm = -INFINITY;
        for (int ww = 0; ww < kWarps; ++ww) mm = fmaxf(mm, sm.m[ww][h]);
        float ll = 0.f, aa = 0.f;
        for (int ww = 0; ww < kWarps; ++ww) {
            if (sm.m[ww][h] == -INFINITY) continue;
            const float f = expf(sm.m[ww][h] - mm);
            ll = fmaf(sm.l[ww][h], f, ll);
            aa = fmaf(sm.acc[ww][h][d], f, aa);
        }
        const size_t slot = ((size_t) r * kHeads + g * kGroup + h) * z + s;
        part_acc[slot * kHeadDim + d] = aa;
        if (d == 0) {
            part_ml[slot * 2] = mm;
            part_ml[slot * 2 + 1] = ll;
        }
    }
}

// A block per (row, head), a thread a channel: the row's slices merged with the sink, gated, written in fp32 and/or as
// q8_1 (a warp a block of 32 channels).
__global__ void __launch_bounds__(kHeadDim) attn_combine_kernel(AttnArgs a, int z, const float* __restrict__ part_ml,
                                                                const float* __restrict__ part_acc, float* __restrict__ out) {
    bk::pdl_wait();
    bk::pdl_launch_dependents();
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int p, lo, n;
    row_keys(a, r, p, lo, n);
    const int span = span_of(n), splits = (n + span - 1) / span;
    const size_t base = ((size_t) r * kHeads + h) * z;
    const float t = a.sink[(size_t) r * kHeads + h];
    float mm = t;
    for (int s = 0; s < splits; ++s) mm = fmaxf(mm, part_ml[(base + s) * 2]);
    float ll = expf(t - mm), aa = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float f = expf(part_ml[(base + s) * 2] - mm);
        ll = fmaf(part_ml[(base + s) * 2 + 1], f, ll);
        aa = fmaf(part_acc[(base + s) * kHeadDim + d], f, aa);
    }
    float o = aa / ll;
    const int c = h * kHeadDim + d;
    if (a.gate) o *= 1.f / (1.f + expf(-a.gate[(size_t) r * a.gate_ld + c]));
    if (out) out[(size_t) r * kHeads * kHeadDim + c] = o;
    if (a.q8) {
        const float amax = warp_max(fabsf(o));
        const float sum = warp_sum(o);
        const float dq = bk::q8_1_finite(amax / 127.0f);
        Q81& b = reinterpret_cast<Q81*>(a.q8)[(size_t) r * (kHeads * kHeadDim / 32) + c / 32];
        b.qs[c % 32] = bk::q8_1_quant(o, dq, amax);
        if ((c & 31) == 0) b.ds = bk::q8_1_ds(dq, sum);
    }
}

// ---- the fused window attention ----------------------------------------------------------------------------------------
//
// A window of up to kFusedRows rows whose keys split in at most kClusterSplits slices of kUnit keys (every sliding-window
// visit; a full visit up to 1024 keys) runs its preparation, its slices and their merge in one kernel: a cluster of
// kClusterSplits blocks per (row, key/value head), block s taking slice s with one tile of kTile keys a warp. Before its
// wait a block reads what no kernel of the window writes: its tile's keys and values of the positions before the window
// (the cache holds them from earlier windows) and their relative-bias rows. It prepares what it reads (prep.cuh) in place
// of prepare_kernel: its three query heads' queries, relative states and sinks, and the keys and values of the window's
// rows whose positions fall in its slice, rounded as the cache stores them and read from its own shared memory; the block
// whose slice holds its own row's position writes that row's into the cache and the rings. The slice then runs as
// attn_split_kernel, and blocks 0..2 of the cluster merge one head each from the slices' shared memory as
// attn_combine_kernel does: the same arithmetic in the same order as the three kernels. A longer window keeps the three:
// each block would prepare up to all of its rows, and a cluster holds its idle slices' blocks.
constexpr int kClusterSplits = 8;
constexpr int kFusedRows = 3;

// ROWS: the window rows a block may prepare (2 for windows of 1-2 rows, 3 for 3)
// One row of a fresh cache row in the cache's format, codes first (fp16: 128 halfs, E4M3/MXFP8: 128 codes) and, for
// MXFP8, the four group scales after them. Fresh rows are staged per block and read from it after the barrier; the
// block whose own row is in its slice writes the same bytes into the cache and the rings.
template <int ROWS, KVFormat F>
struct FusedSmem {
    Smem<F> s;
    alignas(16) uint8_t kw[ROWS][head_code_bytes(F)];
    alignas(16) uint8_t vw[ROWS][head_code_bytes(F)];
    // the window's rows' MXFP8 group scales, [4] per row (fp16/E4M3: unused)
    alignas(4) uint8_t ke[ROWS][F == KVFormat::Mxfp8 ? kMxfp8Groups : 1];
    alignas(4) uint8_t ve[ROWS][F == KVFormat::Mxfp8 ? kMxfp8Groups : 1];
    float pm[kGroup], pl[kGroup];                 // the slice's merged maximum and sum of each head
    float pacc[kGroup][kHeadDim];
};

template <KVFormat F, int ROWS>
__global__ void __cluster_dims__(1, 1, kClusterSplits) __launch_bounds__(kWarps * 32) attn_fused_kernel(AttnPrep pa, AttnArgs a) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    FusedSmem<ROWS, F>& fs = *reinterpret_cast<FusedSmem<ROWS, F>*>(smem_raw);
    Smem<F>& sm = fs.s;
    const int r = blockIdx.x, g = blockIdx.y, s = blockIdx.z;
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int kRowBytes = F == KVFormat::Fp16 ? kHeadDim * 2 : F == KVFormat::Int8 ? kKvInt8RowBytes : kHeadDim;
    constexpr int kLoads = F == KVFormat::Fp16 ? kHeadDim / 8 : kHeadDim / 16;
    // before the wait: the positions (staged before the window), the slice, the warp's tile and what of it is older than
    // the window, and the weights
    int p, lo, n;
    row_keys(a, r, p, lo, n);
    const int span = span_of(n), splits = (n + span - 1) / span;
    const bool active = s < splits;
    const int s_lo = lo + s * span, s_hi = min(p + 1, s_lo + span);
    const int p0 = a.pos[0];   // the window's first position: keys from it on are the window's rows (consecutive positions)
    const int key = s_lo + w * kTile + lane;
    const bool valid = active && key < s_hi;
    const uint8_t* kbase = static_cast<const uint8_t*>(a.k_cache) + (size_t) g * kRowBytes;
    const uint8_t* vbase = static_cast<const uint8_t*>(a.v_cache) + (size_t) g * kRowBytes;
    const uint8_t* kscales = static_cast<const uint8_t*>(a.k_cache) + code_bytes(F) + 4 * g;
    const uint8_t* vscales = static_cast<const uint8_t*>(a.v_cache) + code_bytes(F) + 4 * g;
    uint4 kr[kLoads], vr[kLoads];
    uint8_t ke[4] = {0, 0, 0, 0}, ve[4] = {0, 0, 0, 0};
    float ks[2] = {1.f, 1.f}, vs[2] = {1.f, 1.f};
    if (valid && key < p0) {
        const size_t cell = (size_t) (key % a.cap) * cell_bytes(F);
#pragma unroll
        for (int i = 0; i < kLoads; ++i) kr[i] = reinterpret_cast<const uint4*>(kbase + cell)[i];
#pragma unroll
        for (int i = 0; i < kLoads; ++i) vr[i] = reinterpret_cast<const uint4*>(vbase + cell)[i];
        if constexpr (F == KVFormat::Int8) {
            row_scales(kbase + cell, ks);
            row_scales(vbase + cell, vs);
        } else if constexpr (F == KVFormat::Mxfp8) {
            const uint8_t* ksp = kscales + cell;
            const uint8_t* vsp = vscales + cell;
#pragma unroll
            for (int j = 0; j < 4; ++j) { ke[j] = ksp[j]; ve[j] = vsp[j]; }
        }
    } else {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) kr[i] = vr[i] = make_uint4(0, 0, 0, 0);
    }
    const int dist = p - key;
    float tv[kRel];
#pragma unroll
    for (int i = 0; i < kRel; ++i) tv[i] = valid && dist < kExtent ? a.rel_table[i * kExtent + dist] : 0.f;
    float wt[4][kConv];
    prep::load_weights(pa, g, w, lane, wt);
    bk::pdl_wait();
    // the merging blocks' sink and gate, read early
    float sink = 0.f, gate = 0.f;
    if (s < kGroup) {
        const int h = g * kGroup + s;
        sink = pa.yq[(size_t) r * pa.ldq + kOffSink + h];   // prepare_kernel's sink: the projection as it is
        if (a.gate) gate = a.gate[(size_t) r * a.gate_ld + h * kHeadDim + threadIdx.x];
    }
    if (active) {
        if (w >= 2) {
            float s3[kGroup];
            prep::q_heads(pa, r, p, g, w, lane, wt, &sm.q[0][0], &sm.rs[0][0], s3);
        } else {
            // the window's rows of this slice up to the block's own: both prepared from one round of loads
            const int rr0 = max(0, s_lo - p0), nr = min(r, s_hi - 1 - p0) - rr0 + 1;
            if (nr > 0) {
                float val[ROWS][4], raw[ROWS][4], f[ROWS], scale[ROWS][2] = {};
                prep::kv_rows<ROWS>(pa, rr0, nr, p0 + rr0, g, w, lane, wt, val, raw, f, scale);
#pragma unroll
                for (int k = 0; k < ROWS; ++k) {
                    if (k >= nr) break;
                    const int rr = rr0 + k;
                    float vals[4] = {val[k][0] * f[k], val[k][1] * f[k], val[k][2] * f[k], val[k][3] * f[k]};
                    uint8_t* row = w ? fs.vw[rr] : fs.kw[rr];
                    uint8_t* es = w ? fs.ve[rr] : fs.ke[rr];
                    if constexpr (F == KVFormat::Fp16) {
#pragma unroll
                        for (int u = 0; u < 4; ++u) prep::store_kv<F>(row, lane * 8 + u * 2, vals[u]);
                    } else if constexpr (F == KVFormat::Fp8) {
#pragma unroll
                        for (int u = 0; u < 4; ++u) prep::store_kv<F>(row, lane * 4 + u, vals[u]);
                    } else if constexpr (F == KVFormat::Int8) {
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            prep::store_kv<KVFormat::Int8>(row, (size_t) lane * 4 + u, vals[u], scale[k], lane);
                    } else {
                        mxfp8_encode_row(vals, row, es, lane);
                    }
                    if (rr == r) {   // the block's own row: into the cache and the ring
                        void* cache = w ? pa.v_cache : pa.k_cache;
                        if constexpr (F == KVFormat::Fp16) {
                            const size_t cell = (size_t) (p % pa.cap) * cell_bytes(F) + (size_t) g * head_code_bytes(F);
#pragma unroll
                            for (int u = 0; u < 4; ++u) prep::store_kv<F>(cache, cell + lane * 8 + u * 2, vals[u]);
                        } else if constexpr (F == KVFormat::Fp8) {
                            const size_t cell = (size_t) (p % pa.cap) * cell_bytes(F) + (size_t) g * head_code_bytes(F);
#pragma unroll
                            for (int u = 0; u < 4; ++u) prep::store_kv<F>(cache, cell + lane * 4 + u, vals[u]);
                        } else if constexpr (F == KVFormat::Int8) {
                            const size_t cache_row = (size_t) (p % pa.cap) * kKvHeads * (size_t) kRowBytes + (size_t) g * kRowBytes;
#pragma unroll
                            for (int u = 0; u < 4; ++u)
                                prep::store_kv<KVFormat::Int8>(static_cast<uint8_t*>(cache) + cache_row, (size_t) lane * 4 + u, vals[u], scale[k], lane);
                        } else {   // the fresh row's bytes: the same encode, reused for the cache write
                            const size_t cell = (size_t) (p % pa.cap) * cell_bytes(F) + (size_t) g * head_code_bytes(F);
                            *reinterpret_cast<uint32_t*>((uint8_t*) cache + cell + lane * 4) =
                            *reinterpret_cast<const uint32_t*>(row + lane * 4);
                            if ((lane & 7) == 0) {
                                *((uint8_t*) cache + (size_t) (p % pa.cap) * cell_bytes(F) + code_bytes(F) + 4 * g + (lane >> 3)) = es[lane >> 3];
                            }
                        }
                        float* ring = w ? pa.v_ring : pa.k_ring;
                        *reinterpret_cast<float4*>(ring + (size_t) (p % kConvRing) * (kKvHeads * kHeadDim) + g * kHeadDim + lane * 4) =
                            make_float4(raw[k][0], raw[k][1], raw[k][2], raw[k][3]);
                    }
                }
            }
        }
    }
    __syncthreads();
    // the dependents in only now: post_attn and the output projection read their weights before their waits, and
    // started at the wait they put those reads in front of the preparation's (2-3x its time in the window)
    bk::pdl_launch_dependents();
    if (active) {
        if (valid && key >= p0) {   // a row of this window, as prepared above
            const uint4* kp = reinterpret_cast<const uint4*>(fs.kw[key - p0]);
            const uint4* vp = reinterpret_cast<const uint4*>(fs.vw[key - p0]);
#pragma unroll
            for (int i = 0; i < kLoads; ++i) kr[i] = kp[i];
#pragma unroll
            for (int i = 0; i < kLoads; ++i) vr[i] = vp[i];
            if constexpr (F == KVFormat::Int8) {
                row_scales(fs.kw[key - p0], ks);
                row_scales(fs.vw[key - p0], vs);
            } else if constexpr (F == KVFormat::Mxfp8) {
                const uint8_t* ksp = fs.ke[key - p0];
                const uint8_t* vsp = fs.ve[key - p0];
#pragma unroll
                for (int j = 0; j < 4; ++j) { ke[j] = ksp[j]; ve[j] = vsp[j]; }
            }
        }
        float m[kGroup], l[kGroup], acc[kGroup][4];
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            m[h] = -INFINITY;
            l[h] = 0.f;
#pragma unroll
            for (int u = 0; u < 4; ++u) acc[h][u] = 0.f;
        }
        if (s_lo + w * kTile < s_hi) {   // the warp's tile (attn_split_kernel's loop body, once: a slice is kUnit keys)
            float sc[kGroup] = {0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < kLoads; ++i) {
                if constexpr (F == KVFormat::Fp16) {
                    float f[8];
                    unpack8_f16(kr[i], f);
#pragma unroll
                    for (int d = 0; d < 8; ++d)
#pragma unroll
                        for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[h][8 * i + d], f[d], sc[h]);
                } else {
                    float f[16];
                    if constexpr (F == KVFormat::Fp8) unpack16_fp8(kr[i], f);
                    else if constexpr (F == KVFormat::Int8) unpack16_int8(kr[i], ks[i / 4], f);
                    else mxfp8_decode16(kr[i], mxfp8_scale_factor(ke[i >> 1]), f);
#pragma unroll
                    for (int d = 0; d < 16; ++d)
#pragma unroll
                        for (int h = 0; h < kGroup; ++h) sc[h] = fmaf(sm.q[h][16 * i + d], f[d], sc[h]);
                }
            }
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                float b = 0.f;
#pragma unroll
                for (int i = 0; i < kRel; ++i) b = fmaf(sm.rs[h][i], tv[i], b);
                sc[h] = valid ? sc[h] * a.scale + b : -INFINITY;
            }
            uint32_t* vrow = sm.v[w][lane];
            if constexpr (F == KVFormat::Fp16) {
                for (int i = 0; i < kLoads; ++i) *reinterpret_cast<uint4*>(vrow + 4 * i) = vr[i];
            } else if constexpr (F == KVFormat::Int8) {
#pragma unroll
                for (int i = 0; i < kLoads; ++i) {
                    uint32_t h[8];
                    unpack16_int8_h2(vr[i], __float2half2_rn(vs[i / 4]), h);
#pragma unroll
                    for (int d = 0; d < 8; ++d) vrow[8 * i + d] = h[d];
                }
            } else {   // fp8 and mxfp8
#pragma unroll
                for (int i = 0; i < kLoads; ++i) {
                    float f[16];
                    unpack16_fp8(vr[i], f);
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        const __half2 hv = __floats2half2_rn(f[2 * d], f[2 * d + 1]);
                        vrow[8 * i + d] = *reinterpret_cast<const uint32_t*>(&hv);
                    }
                }
                if constexpr (F == KVFormat::Mxfp8) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) sm.vsc[w][lane][j] = ve[j];
                }
            }
            float pr[kGroup];
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                const float mt = warp_max(sc[h]);
                const float mn = fmaxf(m[h], mt);
                const float corr = m[h] == -INFINITY ? 0.f : expf(m[h] - mn);
                pr[h] = sc[h] == -INFINITY ? 0.f : expf(sc[h] - mn);
                l[h] = l[h] * corr + warp_sum(pr[h]);
#pragma unroll
                for (int u = 0; u < 4; ++u) acc[h][u] *= corr;
                m[h] = mn;
            }
            __syncwarp();
            for (int j = 0; j < kTile; ++j) {
                const uint2 u = *reinterpret_cast<const uint2*>(&sm.v[w][j][lane * 2]);
                float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
                float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
                if constexpr (F == KVFormat::Mxfp8) {
                    const float scale = mxfp8_scale_factor(sm.vsc[w][j][lane >> 3]);
                    v01.x *= scale; v01.y *= scale; v23.x *= scale; v23.y *= scale;
                }
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    const float pj = __shfl_sync(0xffffffffu, pr[h], j);
                    acc[h][0] = fmaf(pj, v01.x, acc[h][0]);
                    acc[h][1] = fmaf(pj, v01.y, acc[h][1]);
                    acc[h][2] = fmaf(pj, v23.x, acc[h][2]);
                    acc[h][3] = fmaf(pj, v23.y, acc[h][3]);
                }
            }
            __syncwarp();
        }
        // the four warps' parts of this slice, merged in warp order
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            if (lane == 0) {
                sm.m[w][h] = m[h];
                sm.l[w][h] = l[h];
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) sm.acc[w][h][lane * 4 + u] = acc[h][u];
        }
        __syncthreads();
        for (int i = threadIdx.x; i < kGroup * kHeadDim; i += blockDim.x) {
            const int h = i / kHeadDim, d = i % kHeadDim;
            float mm = -INFINITY;
            for (int ww = 0; ww < kWarps; ++ww) mm = fmaxf(mm, sm.m[ww][h]);
            float ll = 0.f, aa = 0.f;
            for (int ww = 0; ww < kWarps; ++ww) {
                if (sm.m[ww][h] == -INFINITY) continue;
                const float f = expf(sm.m[ww][h] - mm);
                ll = fmaf(sm.l[ww][h], f, ll);
                aa = fmaf(sm.acc[ww][h][d], f, aa);
            }
            fs.pacc[h][d] = aa;
            if (d == 0) {
                fs.pm[h] = mm;
                fs.pl[h] = ll;
            }
        }
    }
    // the slices' shared memory to the merging blocks: a release and an acquire of shared memory only (a cluster-wide
    // release of every state space is a GPU-scope fence that waits for the cache and ring stores; no block of the cluster
    // reads those)
    asm volatile("fence.release.sync_restrict::shared::cta.cluster;\n\tbarrier.cluster.arrive.relaxed.aligned;\n\t"
                 "barrier.cluster.wait.aligned;\n\tfence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
    if (s < kGroup) {   // head g * 3 + s: the row's slices merged with the sink, gated, rounded to q8_1 (a warp a block of 32)
        const int h = s, d = threadIdx.x;
        cg::cluster_group cluster = cg::this_cluster();
        // every slice's maximum, sum and value read at once, then merged in slice order
        float pm[kClusterSplits], pl[kClusterSplits], pv[kClusterSplits];
#pragma unroll
        for (int q = 0; q < kClusterSplits; ++q)
            if (q < splits) {
                const FusedSmem<ROWS, F>* part = cluster.map_shared_rank(&fs, q);
                pm[q] = part->pm[h];
                pl[q] = part->pl[h];
                pv[q] = part->pacc[h][d];
            }
        float mm = sink;
#pragma unroll
        for (int q = 0; q < kClusterSplits; ++q)
            if (q < splits) mm = fmaxf(mm, pm[q]);
        float ll = expf(sink - mm), aa = 0.f;
#pragma unroll
        for (int q = 0; q < kClusterSplits; ++q)
            if (q < splits) {
                const float f = expf(pm[q] - mm);
                ll = fmaf(pl[q], f, ll);
                aa = fmaf(pv[q], f, aa);
            }
        float o = aa / ll;
        const int c = (g * kGroup + h) * kHeadDim + d;
        if (a.gate) o *= 1.f / (1.f + expf(-gate));
        const float amax = warp_max(fabsf(o));
        const float sum = warp_sum(o);
        const float dq = bk::q8_1_finite(amax / 127.0f);
        Q81& b = reinterpret_cast<Q81*>(a.q8)[(size_t) r * (kHeads * kHeadDim / 32) + c / 32];
        b.qs[c % 32] = bk::q8_1_quant(o, dq, amax);
        if ((c & 31) == 0) b.ds = bk::q8_1_ds(dq, sum);
    }
    // no block leaves while the merging blocks still read its slice: an arrival without release semantics (a release
    // here is a GPU-scope fence that waits for the q8_1 stores to be acknowledged); the merging blocks' remote loads
    // are complete, their values are in the stores above
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
}

int launch_splits(int64_t max_keys) {
    return (int) std::min<int64_t>(kMaxSplits, std::max<int64_t>(1, (max_keys + kUnit - 1) / kUnit));
}

}  // namespace

int opt_in_device() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kOptInDevices) throw std::runtime_error("inkling2: device out of range");
    return dev;
}

size_t attention_decode_scratch(int rows, int64_t max_keys) {
    const size_t slots = (size_t) rows * kHeads * (size_t) launch_splits(max_keys);
    return slots * (2 + kHeadDim) * sizeof(float);
}

void attention_decode(const AttnArgs& a, int rows, int64_t max_keys, void* scratch, float* out, void* stream) {
    if (rows <= 0) return;
    cudaStream_t s = (cudaStream_t) stream;
    const int z = launch_splits(max_keys);
    float* part_ml = static_cast<float*>(scratch);
    float* part_acc = part_ml + (size_t) rows * kHeads * z * 2;
    static bool attr[kOptInDevices] = {};   // the opt-in is per device: a layer split launches these on two cards
    if (bool& set = attr[opt_in_device()]; !set) {
        cudaFuncSetAttribute(attn_split_kernel<KVFormat::Fp16>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<KVFormat::Fp16>));
        cudaFuncSetAttribute(attn_split_kernel<KVFormat::Fp8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<KVFormat::Fp8>));
        cudaFuncSetAttribute(attn_split_kernel<KVFormat::Int8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<KVFormat::Int8>));
        cudaFuncSetAttribute(attn_split_kernel<KVFormat::Mxfp8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(Smem<KVFormat::Mxfp8>));
        set = true;

    }
    const dim3 grid((unsigned) rows, kKvHeads, (unsigned) z);
    switch (a.format) {
    case KVFormat::Fp16:  bk::launch_pdl(attn_split_kernel<KVFormat::Fp16>, grid, dim3(kWarps * 32), sizeof(Smem<KVFormat::Fp16>), s, a, z, part_ml, part_acc); break;
    case KVFormat::Fp8:   bk::launch_pdl(attn_split_kernel<KVFormat::Fp8>, grid, dim3(kWarps * 32), sizeof(Smem<KVFormat::Fp8>), s, a, z, part_ml, part_acc); break;
    case KVFormat::Int8:  bk::launch_pdl(attn_split_kernel<KVFormat::Int8>, grid, dim3(kWarps * 32), sizeof(Smem<KVFormat::Int8>), s, a, z, part_ml, part_acc); break;
    case KVFormat::Mxfp8: bk::launch_pdl(attn_split_kernel<KVFormat::Mxfp8>, grid, dim3(kWarps * 32), sizeof(Smem<KVFormat::Mxfp8>), s, a, z, part_ml, part_acc); break;
    }
    check("attention_decode");
    bk::launch_pdl(attn_combine_kernel, dim3((unsigned) rows, kHeads), dim3(kHeadDim), 0, s, a, z, (const float*) part_ml, (const float*) part_acc, out);
    check("attention_decode/combine");
}

void attention_window(const AttnPrep& pa, const AttnArgs& a, int rows, int64_t max_keys, void* scratch, void* stream) {
    if (rows <= 0) return;
    if (rows > kFusedRows || launch_splits(max_keys) > kClusterSplits || a.q8 == nullptr || a.seg.at.n > 0) {
        decode_prepare(pa, rows, stream);
        attention_decode(a, rows, max_keys, scratch, nullptr, stream);
        return;
    }
    // 1-2 rows, a block an SM: each block asks for more than half of an SM's shared memory (it uses sizeof(FusedSmem)), so
    // a cluster's blocks spread over as many SMs instead of sharing them two by two through the preparation and the tile,
    // while the products' small blocks still fit beside one; 3 rows, two blocks an SM, so the 3 x 8 x 8 blocks take one wave
    static size_t smem_of[kOptInDevices] = {};   // per device, as above
    const int dev = opt_in_device();
    size_t& smem = smem_of[dev];
    if (!smem) {
        int per_sm = 0, reserved = 0;
        cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
        cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev);
        smem = std::max(sizeof(FusedSmem<2, KVFormat::Fp16>), (size_t) std::max(0, per_sm / 2 - reserved + 128));
        cudaFuncSetAttribute(attn_fused_kernel<KVFormat::Fp16, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        cudaFuncSetAttribute(attn_fused_kernel<KVFormat::Fp8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        cudaFuncSetAttribute(attn_fused_kernel<KVFormat::Int8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        cudaFuncSetAttribute(attn_fused_kernel<KVFormat::Mxfp8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        cudaFuncSetAttribute(attn_fused_kernel<KVFormat::Fp16, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(FusedSmem<3, KVFormat::Fp16>));
        cudaFuncSetAttribute(attn_fused_kernel<KVFormat::Fp8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(FusedSmem<3, KVFormat::Fp8>));
        cudaFuncSetAttribute(attn_fused_kernel<KVFormat::Int8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(FusedSmem<3, KVFormat::Int8>));
        cudaFuncSetAttribute(attn_fused_kernel<KVFormat::Mxfp8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(FusedSmem<3, KVFormat::Mxfp8>));
    }
    const dim3 grid((unsigned) rows, kKvHeads, kClusterSplits);
    cudaStream_t s = (cudaStream_t) stream;
    if (rows <= 2) {
        if (a.format == KVFormat::Fp16) bk::launch_pdl(attn_fused_kernel<KVFormat::Fp16, 2>, grid, dim3(kWarps * 32), smem, s, pa, a);
        else if (a.format == KVFormat::Fp8) bk::launch_pdl(attn_fused_kernel<KVFormat::Fp8, 2>, grid, dim3(kWarps * 32), smem, s, pa, a);
        else if (a.format == KVFormat::Int8) bk::launch_pdl(attn_fused_kernel<KVFormat::Int8, 2>, grid, dim3(kWarps * 32), smem, s, pa, a);
        else bk::launch_pdl(attn_fused_kernel<KVFormat::Mxfp8, 2>, grid, dim3(kWarps * 32), smem, s, pa, a);
    } else {
        if (a.format == KVFormat::Fp16) bk::launch_pdl(attn_fused_kernel<KVFormat::Fp16, 3>, grid, dim3(kWarps * 32), sizeof(FusedSmem<3, KVFormat::Fp16>), s, pa, a);
        else if (a.format == KVFormat::Fp8) bk::launch_pdl(attn_fused_kernel<KVFormat::Fp8, 3>, grid, dim3(kWarps * 32), sizeof(FusedSmem<3, KVFormat::Fp8>), s, pa, a);
        else if (a.format == KVFormat::Int8) bk::launch_pdl(attn_fused_kernel<KVFormat::Int8, 3>, grid, dim3(kWarps * 32), sizeof(FusedSmem<3, KVFormat::Int8>), s, pa, a);
        else bk::launch_pdl(attn_fused_kernel<KVFormat::Mxfp8, 3>, grid, dim3(kWarps * 32), sizeof(FusedSmem<3, KVFormat::Mxfp8>), s, pa, a);
    }
    check("attention_window");
}

}  // namespace basalt::models::inkling2
