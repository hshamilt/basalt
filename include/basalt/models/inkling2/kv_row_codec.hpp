// include/basalt/models/inkling2/kv_row_codec.hpp - the MXFP8 cache codec of Inkling 2: E4M3 codes stored with one
// unsigned 8-bit power-of-two scale (UE8M0) per group of 32 values, the release's rule.
//
// The 32 values of one group share a group amax. The exponent byte e is the exponent of round(amax / 448 up to the next
// power of two) (0 when amax / 448 is at most 2^-127, and at most 254), 255 marked when any value at the codec input is
// NaN or infinite. A value's code is x * 2^(127 - e), clamped to +-448 and rounded to E4M3 nearest-even. A group whose
// exponent is 255 stores the canonical zero codes and decodes as quiet NaNs.
//
// A warp encodes one head of one position (128 values, 4 a lane, lane l holding channels 4l..4l+3, group g spanning
// channels 32g..32g+31 = lanes 8g..8g+7), and the decode is the same shape in the readers. The code's decode is
// scalbnf(E4M3(code), int(e) - 127) in fp32 (2^-127 for e = 0, not zero), and a group whose exponent is 255 decodes as
// quiet NaNs.
#pragma once

#include "basalt/models/inkling2/kernels.hpp"

#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>

namespace basalt::models::inkling2 {

constexpr uint8_t kMxfp8NaN = 255;   ///< the exponent byte marking a group with a NaN or infinity at the codec input

/// The biased exponent of `raw` rounded up to the next power of two, clamped to 254; 0 when `raw` is at most 2^-127.
/// `raw` is the group's amax / 448. NaN at the input is the caller's group-wide marker, never reached here.
__device__ __forceinline__ uint8_t mxfp8_exponent(float amax) {
    const float raw = amax * (1.0f / 448.0f);
    if (raw != raw) return kMxfp8NaN;   ///< the group's amax is NaN (defensive; the group check is the real marker)
    if (raw <= 0x1p-127f) return 0;
    const uint32_t bits = __float_as_uint(raw);
    const uint32_t e = (bits + 0x007FFFFFu) & 0x7F800000u;
    return (uint8_t) (e >> 23 < 254u ? e >> 23 : 254u);
}

/// The code of `x` under the group's inverse scale (2^(127 - e)), clamped to +-448 and rounded to E4M3 nearest even.
__device__ __forceinline__ uint8_t mxfp8_code(float x, float inverse) {
    const float v = fminf(fmaxf(x * inverse, -448.0f), 448.0f);
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

/// Encodes the warp's 128 values (4 a lane, lane l holding channels 4l..4l+3) into `codes` [128] and `scales` [4].
/// Non-finite values in the group mark the group: exponent 255 and canonical zero codes.
__device__ __forceinline__ void mxfp8_encode_row(float (&v)[4], uint8_t* __restrict__ codes,
                                                uint8_t* __restrict__ scales, int lane) {
    const bool bad = !isfinite(v[0]) || !isfinite(v[1]) || !isfinite(v[2]) || !isfinite(v[3]);
    uint32_t badv = bad ? 1u : 0u;
    float amax = fmaxf(fabsf(v[0]), fmaxf(fabsf(v[1]), fmaxf(fabsf(v[2]), fabsf(v[3]))));
#pragma unroll
    for (int m = 4; m; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m, 8));
        badv |= __shfl_xor_sync(0xffffffffu, badv, m, 8);
    }
    const int g = lane >> 3;
    uint32_t out = 0;
    if (badv == 0u) {
        const uint8_t e = mxfp8_exponent(amax);
        if ((lane & 7) == 0) scales[g] = e;
        const float inverse = scalbnf(1.0f, 127 - int(e));
#pragma unroll
        for (int u = 0; u < 4; ++u) out |= uint32_t(mxfp8_code(v[u], inverse)) << (8u * u);
    } else {
        if ((lane & 7) == 0) scales[g] = kMxfp8NaN;
    }
    *reinterpret_cast<uint32_t*>(codes + lane * 4) = out;
}

/// The scale of a group's exponent byte: 2^(e - 127), 2^-127 for e = 0 (never zero), and quiet NaN for e = 255.
__device__ __forceinline__ float mxfp8_scale_factor(uint8_t e) {
    if (e == kMxfp8NaN) return __int_as_float(0x7FC00000u);
    return scalbnf(1.0f, int(e) - 127);
}

/// The 16 values of one 16-byte code piece (4 uint32s) as fp32: E4M3 codes (exact in fp16, then fp32) times the group's
/// scale. Codes at byte offset 0 of the piece are values 0..15 of a 32-value group's first half or second half.
__device__ __forceinline__ void mxfp8_decode16(const uint4 u, float scale, float (&f)[16]) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        __nv_fp8x2_e4m3 lo, hi;
        lo.__x = (__nv_fp8x2_storage_t) (w[i] & 0xFFFFu);
        hi.__x = (__nv_fp8x2_storage_t) (w[i] >> 16);
        const float2 a = (float2) lo, b = (float2) hi;
        f[4 * i] = a.x * scale;
        f[4 * i + 1] = a.y * scale;
        f[4 * i + 2] = b.x * scale;
        f[4 * i + 3] = b.y * scale;
    }
}

}  // namespace basalt::models::inkling2
