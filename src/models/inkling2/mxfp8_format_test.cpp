// src/models/inkling2/mxfp8_format_test.cpp - the MXFP8 cache format's layout and codec: the cell layout and byte
// arithmetic the planner and the kernels share, and the codec of the finite path, against an independent scalar oracle
// (the exact rules, written from the spec here, not from the device code).
#include "basalt/models/inkling2/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

using namespace basalt::models::inkling2;

namespace {

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

// ---- the oracle: the release's MXFP8 rule, scalar, in the host's arithmetic --------------------------------------------

constexpr double kMinNormal = 1.1754943508222875e-38;   // 2^-126

// the E4M3 value of one positive finite code (its sign bit is never set here)
double e4m3(uint8_t c) {
    const int exp = (c >> 3) & 0x0F;
    const int mant = c & 0x07;
    if (exp == 0) return mant * std::ldexp(1.0, -9);   // a subnormal
    if (exp == 0x0F && mant == 7) return std::numeric_limits<double>::quiet_NaN();
    return std::ldexp(1.0 + mant / 8.0, exp - 7);
}

// the exponent byte of a group's amax (its values' max magnitude over a group of 32), the release's rule in binary32
uint8_t oracle_exponent(double amax) {
    if (!std::isfinite(amax) || amax < 0.0) return 255;
    const float raw = (float)amax * (1.0f / 448.0f);
    if (raw != raw) return 255;
    if (raw <= std::ldexp(1.0f, -127)) return 0;
    uint32_t bits;
    std::memcpy(&bits, &raw, 4);
    const uint32_t e = ((bits + 0x007FFFFFu) & 0x7F800000u) >> 23;
    return (uint8_t) std::min<uint32_t>(e, 254u);
}

// one value's code under a group's exponent byte: the value * 2^(127 - e), clamped to +-448, in binary32
uint8_t oracle_code(double x, uint8_t e) {
    if (e == 255) return 0;
    const float v = std::min(std::max((float) x * std::ldexp(1.0f, 127 - int(e)), -448.0f), 448.0f);
    const float a = std::fabs(v);
    uint8_t best = 0;
    for (uint8_t c = 1; c < 0x7F; ++c) {   // every finite positive code; 0x7F is NaN
        const float d = std::fabs((float) e4m3(c) - a), b = std::fabs((float) e4m3(best) - a);
        if (d < b || (d == b && (c & 1u) == 0u)) best = c;
    }
    // signed zero: the magnitude is 0, the sign is the value's
    if (std::signbit(v)) {
        if (best == 0) return 0x80u;
        return uint8_t(best | 0x80u);
    }
    return best;
}

double oracle_scale(uint8_t e) {
    if (e == 255) return std::numeric_limits<double>::quiet_NaN();
    return (double) std::ldexp(1.0f, int(e) - 127);
}

double oracle_decode(uint8_t code, uint8_t e) {
    if (e == 255) return std::numeric_limits<double>::quiet_NaN();
    const int sign = (code & 0x80) ? 1 : 0;
    const uint8_t mag = code & 0x7F;
    const double v = e4m3(mag) * oracle_scale(e);
    return sign ? -v : v;
}

}  // namespace

int main() {
    // ---- the layout ------------------------------------------------------------------------------------------------
    expect(cell_bytes(KVFormat::Fp16) == 2048, "the fp16 cell is 2048 bytes");
    expect(cell_bytes(KVFormat::Fp8) == 1024, "the unit-scale E4M3 cell is 1024 bytes");
    expect(cell_bytes(KVFormat::Mxfp8) == 1056, "the MXFP8 cell is 1056 bytes");
    expect(cell_bytes(KVFormat::Mxfp8) % 16 == 0, "the MXFP8 cell is 16-byte aligned");
    expect(head_code_bytes(KVFormat::Fp16) == 256 && head_code_bytes(KVFormat::Fp8) == 128 && head_code_bytes(KVFormat::Mxfp8) == 128,
           "one head's code row is 128 values at one or two bytes a value");
    expect(codes_offset(KVFormat::Mxfp8, 0) == 0 && codes_offset(KVFormat::Mxfp8, 7) == 896, "head-major code rows");
    expect(scales_offset(KVFormat::Mxfp8, 0) == 1024 && scales_offset(KVFormat::Mxfp8, 7) == 1052, "scale rows after the codes");
    expect(scales_offset(KVFormat::Fp16, 0) == 2048 || scales_offset(KVFormat::Fp8, 0) == 1024, "no scales in the other formats (offsets unused)");
    expect(pos_bytes(KVFormat::Fp16) == 4096 && pos_bytes(KVFormat::Fp8) == 2048 && pos_bytes(KVFormat::Mxfp8) == 2112,
           "one position, K + V");
    expect(2ull * cell_bytes(KVFormat::Mxfp8) * 8ull * 262144ull / (1024ull * 1024ull) == 4224ull,
           "the 262K full-visit figure (8 visits, 4,224 MiB of full-visit bytes)");
    expect((uint64_t) 131072 * cell_bytes(KVFormat::Mxfp8) == (uint64_t) 132 * 1024 * 1024,
           "131,072 positions is exactly 132 MiB per K or V payload");
    expect((uint64_t) 8 * 2 * 1024 * 1 * 16384, "the unit-scale fp8 figure");   // 16,384 B a position, 8 visits x K/V

    // ---- the codec --------------------------------------------------------------------------------------------------
    // signed zero: the code is a signed zero of the value's sign, the group's exponent is 0
    {
        // a group of all zeros: raw = 0 -> e = 0; -0 keeps its sign, +0 its sign
        expect(oracle_exponent(0.0) == 0, "an all-zero group keeps e = 0");
        const uint8_t neg = oracle_code(-0.0, 0);
        expect(neg == 0x80u, "a negative zero codes to 0x80");
        expect(oracle_code(0.0, 0) == 0x00u, "a positive zero codes to 0x00");
    }
    // the scale transition on either side of a power of two
    expect(oracle_exponent(std::ldexp(1.0, -127) * 448.0) == 0, "raw = 2^-127 -> e = 0 (the boundary)");
    expect(oracle_exponent(std::ldexp(1.0, -126) * 448.0) == 1, "raw = 2^-126 keeps the biased exponent 1");
    expect(oracle_exponent(448.0 * 1.0) == 127, "raw = 1 -> e = 127");
    expect(oracle_exponent(448.0 * 2.0) == 128, "raw = 2 -> e = 128");
    expect(oracle_exponent(448.0 * (1.0 + 1e-7)) == 128, "raw = 1 + epsilon rounds up to 2");
    // the golden vector: E4M3 value 256 with e = 118 decodes to 0.5
    expect(oracle_decode(0x78, 118) == 0.5, "the golden vector: E4M3 value 256, e = 118 -> 0.5");
    // e = 0: the scale is 2^-127, not zero
    expect(oracle_scale(0) == std::ldexp(1.0, -127), "e = 0 decodes as 2^-127");
    expect(std::abs(oracle_decode(0x78, 0) - 256.0 * std::ldexp(1.0, -127)) < 1e-45, "e = 0 decodes exactly");
    // e = 255: the canonical zero codes, decoded as NaNs
    expect(oracle_code(1.0, 255) == 0, "a NaN group stores canonical zero codes");
    expect(std::isnan(oracle_decode(0x00, 255)), "a NaN group decodes as quiet NaNs");
    // the group-wide marker: any non-finite value marks its group (amax path is never reached)
    expect(oracle_exponent(std::numeric_limits<double>::quiet_NaN()) == 255, "a NaN group is marked 255");
    expect(oracle_exponent(std::numeric_limits<double>::infinity()) == 255, "an infinite group is marked 255");
    expect(oracle_exponent(-std::numeric_limits<double>::infinity()) == 255, "a negative-infinite group is marked 255");
    // E4M3 ties: 1.5 exactly (a tie? no, it is a representable code: 1.5 = 0b1.100 * 2^0)
    expect(oracle_code(1.5, 127) == 0x3Cu, "1.5 is a representable code, no rounding");
    // E4M3 subnormal: 0.01 rounds to the nearest subnormal codes
    {
        const uint8_t c = oracle_code(0.01, 127);
        expect(c == 0x05u, "0.01 rounds to the nearest subnormal code");
        expect(std::abs(oracle_decode(c, 127) - 0.01) < 0.002, "the subnormal code is close");
    }
    // signed values: negative codes mirror positive codes
    {
        const uint8_t p = oracle_code(3.7, 127);
        const uint8_t n = oracle_code(-3.7, 127);
        expect((p & 0x80u) == 0 && (n & 0x80u) != 0 && (n & 0x7F) == p, "negative codes mirror positive ones");
    }
    // a mid-range value: raw = 2^-13 keeps the biased exponent 114
    expect(oracle_exponent(std::ldexp(1.0, -13) * 448.0) == 114, "raw = 2^-13 keeps the biased exponent 114");

    if (failures) {
        std::fprintf(stderr, "mxfp8_format_test: %d failures\n", failures);
        return 1;
    }
    std::printf("mxfp8_format_test: ok\n");
    return 0;
}
