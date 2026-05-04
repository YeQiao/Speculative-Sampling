/*
 * Fused Mamba2-65M single-step forward pass in C++.
 *
 * Eliminates per-layer Python/PyTorch dispatch overhead by running
 * the entire 16-layer forward (minus embed lookup & LM head) in a
 * single C++ call. Supports mixed precision: BF16 weights with FP32
 * compute for accuracy-critical paths (SSM state, norm).
 *
 * Architecture: hidden=512, d_inner=1024, conv_dim=1280, n_heads=16,
 *   head_dim=64, state_size=128, n_groups=1, conv_kernel=4
 *
 * Memory bandwidth breakdown (B=1, FP32):
 *   LM head/embed: 262.7 MB (tied, 128256×512)
 *   Per-layer: in_proj 4.75MB + out_proj 2.10MB + conv 20KB + norms ~4KB ≈ 6.87MB
 *   16 layers: 110 MB
 *   Total per step: ~373 MB reads → memory-bound at ~50 GB/s DDR5
 *
 * With BF16 weights: ~186 MB → near 2x speedup for memory-bound ops.
 */

#include <torch/extension.h>
#include <cmath>
#include <cstring>

#ifdef __AVX512F__
#include <immintrin.h>
#define HAS_AVX512 1
#else
#define HAS_AVX512 0
#endif

#ifdef __AVX512BF16__
#define HAS_AVX512_BF16 1
#else
#define HAS_AVX512_BF16 0
#endif

#ifdef __AVX512VNNI__
#define HAS_AVX512_VNNI 1
#else
#define HAS_AVX512_VNNI 0
#endif

// ============================================================
// Utility: BF16 <-> FP32 conversion
// ============================================================

inline float bf16_to_fp32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(uint32_t));
    return static_cast<uint16_t>((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}

// ============================================================
// Softplus
// ============================================================
inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return std::log(1.0f + std::exp(x));
}

// ============================================================
// GEMV: y = x @ W^T + bias (W in BF16, x/y in FP32)
// Shapes: x[K], W[M, K], bias[M] -> y[M]
// ============================================================
static void gemv_bf16(
    float* __restrict__ y,
    const float* __restrict__ x,
    const uint16_t* __restrict__ W,  // [M, K] in BF16 row-major
    const float* __restrict__ bias,  // [M] in FP32, or nullptr
    int M, int K
) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const uint16_t* w_row = W + (int64_t)m * K;
        float acc = (bias != nullptr) ? bias[m] : 0.0f;

#if HAS_AVX512
        __m512 v_acc = _mm512_setzero_ps();
        int k = 0;

#if HAS_AVX512_BF16
        // Use native BF16 dot product: vdpbf16ps
        // Processes 32 BF16 pairs → 16 FP32 accumulators per instruction
        for (; k + 32 <= K; k += 32) {
            __m512 v_x_lo = _mm512_loadu_ps(x + k);
            __m512 v_x_hi = _mm512_loadu_ps(x + k + 16);
            // Pack two FP32 vectors into one BF16 register
            __m512bh v_x_bf16 = _mm512_cvtne2ps_pbh(v_x_hi, v_x_lo);
            __m512bh v_w_bf16 = (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + k));
            v_acc = _mm512_dpbf16_ps(v_acc, v_x_bf16, v_w_bf16);
        }
#endif
        // AVX-512 fallback: load BF16, convert to FP32, FMA
        for (; k + 16 <= K; k += 16) {
            __m512 v_x = _mm512_loadu_ps(x + k);
            // Load 16 BF16 values, convert to FP32
            __m256i v_w_i16 = _mm256_loadu_si256((__m256i*)(w_row + k));
            __m512i v_w_i32 = _mm512_cvtepu16_epi32(v_w_i16);
            __m512 v_w = _mm512_castsi512_ps(_mm512_slli_epi32(v_w_i32, 16));
            v_acc = _mm512_fmadd_ps(v_x, v_w, v_acc);
        }

        acc += _mm512_reduce_add_ps(v_acc);

        // Scalar remainder
        for (; k < K; k++) {
            acc += x[k] * bf16_to_fp32(w_row[k]);
        }
#else
        for (int k = 0; k < K; k++) {
            acc += x[k] * bf16_to_fp32(w_row[k]);
        }
#endif
        y[m] = acc;
    }
}

// ============================================================
// GEMV: y = x @ W^T + bias (W in INT8, x/y in FP32)
// Weight-only INT8 quantization with per-row symmetric scaling.
//
// Uses AVX-512 VNNI (vpdpbusd) for native INT8 dot products:
//   - Weights stored as INT8 (signed, per-row scale)
//   - Input quantized to UINT8 on-the-fly (per-vector scale)
//   - Accumulate in INT32, then dequant to FP32
//   - Bias correction for uint8 shift: subtract 128*row_sum*x_scale
//
// Memory: 1 byte/weight vs 2 bytes (BF16) = 2x bandwidth reduction
// Shapes: x[K], W[M, K], scale[M], row_sum[M], bias[M] -> y[M]
// ============================================================
static void gemv_int8(
    float* __restrict__ y,
    const float* __restrict__ x,       // [K] FP32 input
    const int8_t* __restrict__ W,      // [M, K] INT8 weights (signed)
    const float* __restrict__ w_scale, // [M] per-row scale (absmax/127)
    const int32_t* __restrict__ w_row_sum, // [M] sum of int8 values per row
    const float* __restrict__ bias,    // [M] FP32 bias, or nullptr
    int M, int K,
    float x_scale,                     // pre-computed: max(|x|) / 127
    const uint8_t* __restrict__ x_u8   // [K] pre-quantized input as uint8
) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t* w_row = W + (int64_t)m * K;
        int32_t acc_i32 = 0;

#if HAS_AVX512 && HAS_AVX512_VNNI
        __m512i v_acc = _mm512_setzero_si512();
        int k = 0;

        // VNNI: processes 64 uint8×int8 pairs per instruction (16 lanes × 4 pairs)
        for (; k + 64 <= K; k += 64) {
            __m512i v_x = _mm512_loadu_si512((__m512i*)(x_u8 + k));
            __m512i v_w = _mm512_loadu_si512((__m512i*)(w_row + k));
            v_acc = _mm512_dpbusd_epi32(v_acc, v_x, v_w);
        }

        // Reduce 16 int32 lanes to scalar
        acc_i32 = _mm512_reduce_add_epi32(v_acc);

        // Handle remainder with scalar
        for (; k < K; k++) {
            acc_i32 += (int32_t)x_u8[k] * (int32_t)w_row[k];
        }
#elif HAS_AVX512
        // Fallback: load INT8, convert to FP32, FMA (still 2x bandwidth savings)
        __m512 v_acc_f = _mm512_setzero_ps();
        int k = 0;
        for (; k + 16 <= K; k += 16) {
            __m512 v_x = _mm512_loadu_ps(x + k);
            // Load 16 int8 values, sign-extend to int32, convert to float
            __m128i v_w_i8 = _mm_loadu_si128((__m128i*)(w_row + k));
            __m512i v_w_i32 = _mm512_cvtepi8_epi32(v_w_i8);
            __m512 v_w_f = _mm512_cvtepi32_ps(v_w_i32);
            v_acc_f = _mm512_fmadd_ps(v_x, v_w_f, v_acc_f);
        }
        float acc_f = _mm512_reduce_add_ps(v_acc_f);
        for (; k < K; k++) {
            acc_f += x[k] * (float)w_row[k];
        }
        // Dequant: multiply by w_scale (result is already in original scale)
        y[m] = acc_f * w_scale[m] + ((bias != nullptr) ? bias[m] : 0.0f);
        continue;  // skip the VNNI dequant path below
#else
        for (int k = 0; k < K; k++) {
            acc_i32 += (int32_t)x_u8[k] * (int32_t)w_row[k];
        }
#endif

#if HAS_AVX512_VNNI || !HAS_AVX512
        // Dequant: y = (acc - 128*row_sum) * w_scale * x_scale + bias
        // Because x_u8 = x_s8 + 128, dot(x_u8, w) = dot(x_s8, w) + 128*sum(w)
        float result = ((float)acc_i32 - 128.0f * (float)w_row_sum[m]) * w_scale[m] * x_scale;
        if (bias != nullptr) result += bias[m];
        y[m] = result;
#endif
    }
}

// Helper: quantize FP32 vector to UINT8 (symmetric, with +128 offset)
// Returns the scale factor. x_u8[i] = round(x[i] / scale) + 128
static float quantize_input_to_uint8(
    uint8_t* __restrict__ x_u8,
    const float* __restrict__ x,
    int K
) {
    // Find absmax
    float absmax = 0.0f;
#if HAS_AVX512
    __m512 v_max = _mm512_setzero_ps();
    __m512 v_sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        __m512 v = _mm512_loadu_ps(x + k);
        v_max = _mm512_max_ps(v_max, _mm512_and_ps(v, v_sign_mask));
    }
    absmax = _mm512_reduce_max_ps(v_max);
    for (; k < K; k++) {
        float a = std::fabs(x[k]);
        if (a > absmax) absmax = a;
    }
#else
    for (int k = 0; k < K; k++) {
        float a = std::fabs(x[k]);
        if (a > absmax) absmax = a;
    }
#endif

    if (absmax < 1e-10f) {
        std::memset(x_u8, 128, K);  // all zeros in signed space
        return 1e-10f;
    }

    float scale = absmax / 127.0f;
    float inv_scale = 127.0f / absmax;

#if HAS_AVX512
    __m512 v_inv_scale = _mm512_set1_ps(inv_scale);
    __m512i v_128 = _mm512_set1_epi32(128);
    k = 0;
    for (; k + 16 <= K; k += 16) {
        __m512 v = _mm512_loadu_ps(x + k);
        // round(x * inv_scale) + 128, clamp to [0, 255]
        __m512i v_i32 = _mm512_cvt_roundps_epi32(
            _mm512_mul_ps(v, v_inv_scale),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v_i32 = _mm512_add_epi32(v_i32, v_128);
        // Clamp and pack to uint8
        v_i32 = _mm512_max_epi32(v_i32, _mm512_setzero_si512());
        v_i32 = _mm512_min_epi32(v_i32, _mm512_set1_epi32(255));
        // Pack 16 int32 → 16 uint8 (via int16 intermediate)
        __m256i v_i16 = _mm512_cvtepi32_epi16(v_i32);  // 16 x int16
        __m128i v_u8 = _mm256_cvtepi16_epi8(v_i16);    // 16 x int8/uint8
        _mm_storeu_si128((__m128i*)(x_u8 + k), v_u8);
    }
    for (; k < K; k++) {
        int val = (int)std::round(x[k] * inv_scale) + 128;
        x_u8[k] = (uint8_t)std::max(0, std::min(255, val));
    }
#else
    for (int k = 0; k < K; k++) {
        int val = (int)std::round(x[k] * inv_scale) + 128;
        x_u8[k] = (uint8_t)std::max(0, std::min(255, val));
    }
#endif
    return scale;
}

// FP32 GEMV for small matrices where BF16 overhead isn't worth it
static void gemv_fp32(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ W,  // [M, K] row-major
    const float* __restrict__ bias,
    int M, int K
) {
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* w_row = W + (int64_t)m * K;
        float acc = (bias != nullptr) ? bias[m] : 0.0f;

#if HAS_AVX512
        __m512 v_acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 16 <= K; k += 16) {
            __m512 v_x = _mm512_loadu_ps(x + k);
            __m512 v_w = _mm512_loadu_ps(w_row + k);
            v_acc = _mm512_fmadd_ps(v_x, v_w, v_acc);
        }
        acc += _mm512_reduce_add_ps(v_acc);
        for (; k < K; k++) {
            acc += x[k] * w_row[k];
        }
#else
        for (int k = 0; k < K; k++) {
            acc += x[k] * w_row[k];
        }
#endif
        y[m] = acc;
    }
}

// ============================================================
// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
// ============================================================
static void rms_norm(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    int dim, float eps = 1e-5f
) {
    float sum_sq = 0.0f;
#if HAS_AVX512
    __m512 v_sum = _mm512_setzero_ps();
    int d = 0;
    for (; d + 16 <= dim; d += 16) {
        __m512 v_x = _mm512_loadu_ps(x + d);
        v_sum = _mm512_fmadd_ps(v_x, v_x, v_sum);
    }
    sum_sq = _mm512_reduce_add_ps(v_sum);
    for (; d < dim; d++) sum_sq += x[d] * x[d];
#else
    for (int d = 0; d < dim; d++) sum_sq += x[d] * x[d];
#endif
    float rms = 1.0f / std::sqrt(sum_sq / dim + eps);

#if HAS_AVX512
    __m512 v_rms = _mm512_set1_ps(rms);
    d = 0;
    for (; d + 16 <= dim; d += 16) {
        __m512 v_x = _mm512_loadu_ps(x + d);
        __m512 v_w = _mm512_loadu_ps(weight + d);
        _mm512_storeu_ps(out + d, _mm512_mul_ps(_mm512_mul_ps(v_x, v_rms), v_w));
    }
    for (; d < dim; d++) out[d] = x[d] * rms * weight[d];
#else
    for (int d = 0; d < dim; d++) out[d] = x[d] * rms * weight[d];
#endif
}

// ============================================================
// RMSNorm with SiLU gate: out = norm(x * silu(gate)) * weight
// Matches HF MambaRMSNormGated
// ============================================================
static void rms_norm_gated(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ gate,
    const float* __restrict__ weight,
    int dim, float eps = 1e-5f
) {
    // First: compute x * silu(gate) into out (temp buffer)
    float sum_sq = 0.0f;

#if HAS_AVX512
    __m512 v_one = _mm512_set1_ps(1.0f);
    __m512 v_sum = _mm512_setzero_ps();
    int d = 0;
    for (; d + 16 <= dim; d += 16) {
        __m512 v_x = _mm512_loadu_ps(x + d);
        __m512 v_g = _mm512_loadu_ps(gate + d);
        // silu(g) = g / (1 + exp(-g))
        // Use approximation: sigmoid(-g) via exp
        __m512 v_neg_g = _mm512_sub_ps(_mm512_setzero_ps(), v_g);
        // exp(-g) — use _mm512_exp_ps if available via SVML, otherwise scalar loop
        // For portability, use scalar exp in a temp buffer
        float tmp_exp[16];
        _mm512_storeu_ps(tmp_exp, v_neg_g);
        for (int i = 0; i < 16; i++) tmp_exp[i] = std::exp(tmp_exp[i]);
        __m512 v_exp = _mm512_loadu_ps(tmp_exp);
        __m512 v_sigmoid = _mm512_div_ps(v_one, _mm512_add_ps(v_one, v_exp));
        __m512 v_silu = _mm512_mul_ps(v_g, v_sigmoid);
        __m512 v_val = _mm512_mul_ps(v_x, v_silu);
        _mm512_storeu_ps(out + d, v_val);
        v_sum = _mm512_fmadd_ps(v_val, v_val, v_sum);
    }
    sum_sq = _mm512_reduce_add_ps(v_sum);
    for (; d < dim; d++) {
        float g = gate[d];
        float silu_g = g / (1.0f + std::exp(-g));
        float val = x[d] * silu_g;
        out[d] = val;
        sum_sq += val * val;
    }
#else
    for (int d = 0; d < dim; d++) {
        float g = gate[d];
        float silu_g = g / (1.0f + std::exp(-g));
        float val = x[d] * silu_g;
        out[d] = val;
        sum_sq += val * val;
    }
#endif

    float rms = 1.0f / std::sqrt(sum_sq / dim + eps);

#if HAS_AVX512
    __m512 v_rms = _mm512_set1_ps(rms);
    d = 0;
    for (; d + 16 <= dim; d += 16) {
        __m512 v_val = _mm512_loadu_ps(out + d);
        __m512 v_w = _mm512_loadu_ps(weight + d);
        _mm512_storeu_ps(out + d, _mm512_mul_ps(_mm512_mul_ps(v_val, v_rms), v_w));
    }
    for (; d < dim; d++) out[d] = out[d] * rms * weight[d];
#else
    for (int d = 0; d < dim; d++) out[d] = out[d] * rms * weight[d];
#endif
}

// ============================================================
// SiLU activation in-place (vectorized with fast exp approx)
// ============================================================
// Fast exp approximation (Schraudolph's method, ~1% relative error)
// Good enough for SiLU where we just need sigmoid shape
inline float fast_exp(float x) {
    // Clamp to avoid overflow/underflow
    if (x > 88.0f) return INFINITY;
    if (x < -88.0f) return 0.0f;
    return std::exp(x);
}

#if HAS_AVX512
// Vectorized SiLU: x * sigmoid(x) = x / (1 + exp(-x))
// Uses a rational polynomial approximation of sigmoid for AVX-512
static inline __m512 _mm512_silu_ps(__m512 x) {
    // sigmoid(x) ≈ 0.5 * (1 + tanh(x * 0.7978845608 * (1 + 0.044715 * x^2)))
    // But simpler: use piecewise linear + polynomial
    // For production quality, just store to temp and use scalar exp
    // The win is from vectorizing everything EXCEPT exp
    float tmp[16];
    _mm512_storeu_ps(tmp, x);
    for (int i = 0; i < 16; i++) {
        tmp[i] = tmp[i] / (1.0f + fast_exp(-tmp[i]));
    }
    return _mm512_loadu_ps(tmp);
}
#endif

static void silu_inplace(float* __restrict__ x, int n) {
    int i = 0;
#if HAS_AVX512
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i);
        _mm512_storeu_ps(x + i, _mm512_silu_ps(v));
    }
#endif
    for (; i < n; i++) {
        x[i] = x[i] / (1.0f + fast_exp(-x[i]));
    }
}

// ============================================================
// Conv1d cached step: TRANSPOSED layout [K, conv_dim]
//
// Key optimization: By storing conv_states as [K, conv_dim] instead
// of [conv_dim, K], we get contiguous memory access for:
//   - Shift: memcpy whole rows (1280 floats contiguous)
//   - Dot product: AVX-512 FMA over 16 channels at once
//   - Bias add + SiLU: AVX-512 vectorized
//
// Layout: conv_state[k * conv_dim + c] = state for channel c at position k
//         conv_weight[k * conv_dim + c] = weight for channel c at position k
// ============================================================
static void conv_silu_step(
    float* __restrict__ out,           // [conv_dim]
    float* __restrict__ conv_state,    // [K, conv_dim] (transposed!)
    const float* __restrict__ new_input, // [conv_dim]
    const float* __restrict__ conv_weight, // [K, conv_dim] (transposed!)
    const float* __restrict__ conv_bias,   // [conv_dim] or nullptr
    int conv_dim, int K
) {
    // Shift rows left: row[0] ← row[1], row[1] ← row[2], ...
    // Each row is conv_dim contiguous floats → fast memcpy
    for (int k = 0; k < K - 1; k++) {
        std::memcpy(conv_state + k * conv_dim,
                    conv_state + (k + 1) * conv_dim,
                    conv_dim * sizeof(float));
    }
    // Append new input to last row
    std::memcpy(conv_state + (K - 1) * conv_dim, new_input, conv_dim * sizeof(float));

    // Dot product over K + bias + SiLU, vectorized over channels
    int c = 0;
#if HAS_AVX512
    for (; c + 16 <= conv_dim; c += 16) {
        __m512 v_acc = _mm512_setzero_ps();

        // K=4 unrolled: 4 FMA operations
        for (int k = 0; k < K; k++) {
            __m512 v_s = _mm512_loadu_ps(conv_state + k * conv_dim + c);
            __m512 v_w = _mm512_loadu_ps(conv_weight + k * conv_dim + c);
            v_acc = _mm512_fmadd_ps(v_s, v_w, v_acc);
        }

        // Add bias
        if (conv_bias) {
            __m512 v_b = _mm512_loadu_ps(conv_bias + c);
            v_acc = _mm512_add_ps(v_acc, v_b);
        }

        // SiLU: x / (1 + exp(-x))
        _mm512_storeu_ps(out + c, _mm512_silu_ps(v_acc));
    }
#endif
    // Scalar remainder (conv_dim=1280 is 16-aligned, so this rarely executes)
    for (; c < conv_dim; c++) {
        float val = 0.0f;
        for (int k = 0; k < K; k++) {
            val += conv_state[k * conv_dim + c] * conv_weight[k * conv_dim + c];
        }
        if (conv_bias) val += conv_bias[c];
        out[c] = val / (1.0f + fast_exp(-val));
    }
}

// ============================================================
// SSM step with AVX-512 (same as ssm_ops.cpp but inline)
// ============================================================
static void ssm_step(
    float* __restrict__ y_out,      // [H, D]
    float* __restrict__ h_new,      // [H, D, N]
    const float* __restrict__ x,    // [H, D]
    const float* __restrict__ B_ssm, // [H, N]
    const float* __restrict__ C_ssm, // [H, N]
    const float* __restrict__ dt,   // [H, D]
    const float* __restrict__ A,    // [H]
    const float* __restrict__ D_skip, // [H]
    const float* __restrict__ h_old, // [H, D, N]
    const float* __restrict__ dt_bias, // [H]
    float time_step_lo, float time_step_hi,
    int H, int D, int N
) {
    for (int h = 0; h < H; h++) {
        for (int d = 0; d < D; d++) {
            int hd = h * D + d;
            float dt_val = dt[hd] + dt_bias[h];
            dt_val = softplus(dt_val);
            if (dt_val < time_step_lo) dt_val = time_step_lo;
            if (dt_val > time_step_hi) dt_val = time_step_hi;

            float dA_val = std::exp(dt_val * A[h]);
            float x_val = x[hd];
            float dBx_scale = dt_val * x_val;

            const float* B_h = B_ssm + h * N;
            const float* C_h = C_ssm + h * N;
            const float* h_old_hd = h_old + (int64_t)hd * N;
            float* h_new_hd = h_new + (int64_t)hd * N;

            float y_acc = 0.0f;

#if HAS_AVX512
            __m512 v_dA = _mm512_set1_ps(dA_val);
            __m512 v_dBx = _mm512_set1_ps(dBx_scale);
            __m512 v_y = _mm512_setzero_ps();

            int n = 0;
            for (; n + 16 <= N; n += 16) {
                __m512 v_ho = _mm512_loadu_ps(h_old_hd + n);
                __m512 v_B = _mm512_loadu_ps(B_h + n);
                __m512 v_C = _mm512_loadu_ps(C_h + n);
                __m512 v_hn = _mm512_fmadd_ps(v_ho, v_dA, _mm512_mul_ps(v_dBx, v_B));
                _mm512_storeu_ps(h_new_hd + n, v_hn);
                v_y = _mm512_fmadd_ps(v_C, v_hn, v_y);
            }
            y_acc = _mm512_reduce_add_ps(v_y);
            for (; n < N; n++) {
                float hn = h_old_hd[n] * dA_val + dBx_scale * B_h[n];
                h_new_hd[n] = hn;
                y_acc += C_h[n] * hn;
            }
#else
            for (int n = 0; n < N; n++) {
                float hn = h_old_hd[n] * dA_val + dBx_scale * B_h[n];
                h_new_hd[n] = hn;
                y_acc += C_h[n] * hn;
            }
#endif
            y_acc += D_skip[h] * x_val;
            y_out[hd] = y_acc;
        }
    }
}

// ============================================================
// Fused full Mamba2 single-step forward (all layers)
//
// Returns logits [vocab_size] for B=1
// ============================================================
torch::Tensor fused_mamba2_step(
    int token_id,
    // Shared embed/LM-head weight (tied) — BF16 [vocab_size, hidden]
    torch::Tensor embed_weight_bf16,
    // Final norm weight — FP32 [hidden]
    torch::Tensor norm_f_weight,
    // Per-layer weights (all FP32 except in_proj/out_proj which are BF16)
    // Packed into flat tensors for minimal Python-side overhead
    torch::Tensor layer_norm_weights,     // [n_layers, hidden]
    torch::Tensor layer_in_proj_bf16,     // [n_layers, proj_size, hidden] BF16
    torch::Tensor layer_in_proj_bias,     // [n_layers, proj_size] FP32
    torch::Tensor layer_conv_weight,      // [n_layers, K, conv_dim] FP32 (transposed for vectorized access)
    torch::Tensor layer_conv_bias,        // [n_layers, conv_dim] FP32
    torch::Tensor layer_A_log,            // [n_layers, H]
    torch::Tensor layer_D,               // [n_layers, H]
    torch::Tensor layer_dt_bias,          // [n_layers, H]
    torch::Tensor layer_ssm_norm_weight,  // [n_layers, d_inner]
    torch::Tensor layer_out_proj_bf16,    // [n_layers, hidden, d_inner] BF16
    torch::Tensor layer_out_proj_bias,    // [n_layers, hidden] FP32
    // Cache state (modified in-place)
    torch::Tensor conv_states,    // [n_layers, K, conv_dim] (transposed layout)
    torch::Tensor ssm_states,     // [n_layers, H, D, N]
    // Architecture dims
    int64_t n_layers,
    int64_t hidden_size,
    int64_t d_inner,
    int64_t conv_dim,
    int64_t n_heads,
    int64_t head_dim,
    int64_t state_size,
    int64_t n_groups,
    int64_t conv_kernel,
    float time_step_lo,
    float time_step_hi,
    // Projection layout
    int64_t d_mlp,
    int64_t proj_size,
    // Optional guidance deltas — FP32 [n_layers, d_inner] or empty
    // Added to x-branch of hbc (first d_inner elements) after in_proj.
    // Computed per-round from verifier hidden states. Same delta reused
    // for all draft steps in a round.
    torch::Tensor guidance_deltas
) {
    const int H = n_heads;
    const int D = head_dim;
    const int N = state_size;
    const int K = conv_kernel;
    const int vocab_size = embed_weight_bf16.size(0);

    // Working buffers (stack-allocated for B=1)
    // Max size needed: proj_size (2320 for Mamba2-65M)
    std::vector<float> hidden(hidden_size);
    std::vector<float> normed(hidden_size);
    std::vector<float> projected(proj_size);
    std::vector<float> conv_out(conv_dim);
    std::vector<float> ssm_y(H * D);
    std::vector<float> ssm_h_new(H * D * N);
    std::vector<float> gated_out(d_inner);
    std::vector<float> layer_out(hidden_size);

    // 1. Embedding lookup (BF16 → FP32)
    const uint16_t* embed_ptr = embed_weight_bf16.data_ptr<at::BFloat16>() != nullptr
        ? reinterpret_cast<const uint16_t*>(embed_weight_bf16.data_ptr<at::BFloat16>())
        : nullptr;
    if (embed_ptr) {
        const uint16_t* row = embed_ptr + (int64_t)token_id * hidden_size;
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] = bf16_to_fp32(row[i]);
        }
    }

    // Pointers
    float* norm_f_ptr = norm_f_weight.data_ptr<float>();
    float* ln_w_ptr = layer_norm_weights.data_ptr<float>();
    const uint16_t* in_proj_ptr = reinterpret_cast<const uint16_t*>(
        layer_in_proj_bf16.data_ptr<at::BFloat16>());
    float* in_proj_bias_ptr = layer_in_proj_bias.data_ptr<float>();
    float* conv_w_ptr = layer_conv_weight.data_ptr<float>();
    float* conv_b_ptr = layer_conv_bias.data_ptr<float>();
    float* alog_ptr = layer_A_log.data_ptr<float>();
    float* d_ptr = layer_D.data_ptr<float>();
    float* dtb_ptr = layer_dt_bias.data_ptr<float>();
    float* ssm_nw_ptr = layer_ssm_norm_weight.data_ptr<float>();
    const uint16_t* out_proj_ptr = reinterpret_cast<const uint16_t*>(
        layer_out_proj_bf16.data_ptr<at::BFloat16>());
    float* out_proj_bias_ptr = layer_out_proj_bias.data_ptr<float>();
    float* cs_ptr = conv_states.data_ptr<float>();
    float* ss_ptr = ssm_states.data_ptr<float>();

    // Strides
    const int64_t in_proj_layer_stride = (int64_t)proj_size * hidden_size;
    const int64_t out_proj_layer_stride = (int64_t)hidden_size * d_inner;
    const int64_t conv_w_layer_stride = (int64_t)conv_dim * K;
    const int64_t cs_layer_stride = (int64_t)conv_dim * K;
    const int64_t ss_layer_stride = (int64_t)H * D * N;

    // Guidance delta pointer (may be null if no guidance)
    const float* guide_ptr = (guidance_deltas.defined() && guidance_deltas.numel() > 0)
        ? guidance_deltas.data_ptr<float>() : nullptr;

    // 2. Run through all layers
    for (int layer = 0; layer < n_layers; layer++) {
        // RMSNorm
        rms_norm(normed.data(), hidden.data(),
                 ln_w_ptr + layer * hidden_size, hidden_size);

        // in_proj (BF16 GEMV)
        gemv_bf16(projected.data(), normed.data(),
                  in_proj_ptr + layer * in_proj_layer_stride,
                  in_proj_bias_ptr + layer * proj_size,
                  proj_size, hidden_size);

        // Split: [d_mlp, d_mlp, gate(d_inner), hbc(conv_dim), dt(H)]
        float* gate = projected.data() + 2 * d_mlp;
        float* hbc = gate + d_inner;
        float* dt = hbc + conv_dim;

        // Inject guidance delta into x-branch (first d_inner of hbc)
        if (guide_ptr) {
            const float* layer_delta = guide_ptr + layer * d_inner;
            for (int i = 0; i < d_inner; i++) {
                hbc[i] += layer_delta[i];
            }
        }

        // Conv1d cached step + SiLU
        conv_silu_step(conv_out.data(),
                       cs_ptr + layer * cs_layer_stride,
                       hbc, conv_w_ptr + layer * conv_w_layer_stride,
                       conv_b_ptr + layer * conv_dim,
                       conv_dim, K);

        // Split conv output: x(d_inner) + B(n_groups*N) + C(n_groups*N)
        float* x_ssm = conv_out.data();
        float* B_raw = x_ssm + d_inner;
        float* C_raw = B_raw + n_groups * N;

        // Reshape B, C: [n_groups, N] → [H, N] by repeating groups
        // For n_groups=1, this is just broadcasting. For n_groups>1, need expansion.
        std::vector<float> B_expanded(H * N);
        std::vector<float> C_expanded(H * N);
        int heads_per_group = H / n_groups;
        for (int g = 0; g < n_groups; g++) {
            for (int hg = 0; hg < heads_per_group; hg++) {
                int h_idx = g * heads_per_group + hg;
                std::memcpy(B_expanded.data() + h_idx * N, B_raw + g * N, N * sizeof(float));
                std::memcpy(C_expanded.data() + h_idx * N, C_raw + g * N, N * sizeof(float));
            }
        }

        // Prepare dt: x_ssm is [d_inner=H*D], dt is [H]
        // Expand dt[H] → [H, D] by broadcasting
        std::vector<float> dt_expanded(H * D);
        for (int h = 0; h < H; h++) {
            for (int d = 0; d < D; d++) {
                dt_expanded[h * D + d] = dt[h];
            }
        }

        // Compute A = -exp(A_log)
        std::vector<float> A_neg(H);
        for (int h = 0; h < H; h++) {
            A_neg[h] = -std::exp(alog_ptr[layer * H + h]);
        }

        // SSM step
        ssm_step(ssm_y.data(), ssm_h_new.data(),
                 x_ssm, B_expanded.data(), C_expanded.data(),
                 dt_expanded.data(), A_neg.data(),
                 d_ptr + layer * H,
                 ss_ptr + layer * ss_layer_stride,
                 dtb_ptr + layer * H,
                 time_step_lo, time_step_hi,
                 H, D, N);

        // Copy new state back
        std::memcpy(ss_ptr + layer * ss_layer_stride, ssm_h_new.data(),
                     H * D * N * sizeof(float));

        // Gated RMSNorm: norm(y * silu(gate)) * weight
        rms_norm_gated(gated_out.data(), ssm_y.data(), gate,
                       ssm_nw_ptr + layer * d_inner, d_inner);

        // out_proj (BF16 GEMV): [hidden, d_inner] @ [d_inner] → [hidden]
        gemv_bf16(layer_out.data(), gated_out.data(),
                  out_proj_ptr + layer * out_proj_layer_stride,
                  out_proj_bias_ptr + layer * hidden_size,
                  hidden_size, d_inner);

        // Residual add
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] += layer_out[i];
        }
    }

    // 3. Final RMSNorm
    rms_norm(normed.data(), hidden.data(), norm_f_ptr, hidden_size);

    // 4. LM head (BF16 GEMV): [vocab_size, hidden] @ [hidden] → [vocab_size]
    auto logits = torch::empty({1, vocab_size}, torch::kFloat32);
    gemv_bf16(logits.data_ptr<float>(), normed.data(),
              embed_ptr, nullptr, vocab_size, hidden_size);

    return logits;
}


// ============================================================
// INT8 Fused Forward: same as above but with INT8 weight-only
// quantization for embed/LM-head, in_proj, and out_proj.
//
// Memory: ~93 MB per step (vs 186 MB BF16) = 2x bandwidth reduction
// Uses AVX-512 VNNI for native INT8 dot products.
// ============================================================
torch::Tensor fused_mamba2_step_int8(
    int token_id,
    // Embed weight — INT8 [vocab_size, hidden] + scale [vocab_size]
    torch::Tensor embed_weight_int8,
    torch::Tensor embed_scale,
    torch::Tensor embed_row_sum,
    // Final norm weight — FP32 [hidden]
    torch::Tensor norm_f_weight,
    // Per-layer weights
    torch::Tensor layer_norm_weights,     // [n_layers, hidden] FP32
    torch::Tensor layer_in_proj_int8,     // [n_layers, proj_size, hidden] INT8
    torch::Tensor layer_in_proj_scale,    // [n_layers, proj_size] FP32
    torch::Tensor layer_in_proj_row_sum,  // [n_layers, proj_size] INT32
    torch::Tensor layer_in_proj_bias,     // [n_layers, proj_size] FP32
    torch::Tensor layer_conv_weight,      // [n_layers, K, conv_dim] FP32
    torch::Tensor layer_conv_bias,        // [n_layers, conv_dim] FP32
    torch::Tensor layer_A_log,            // [n_layers, H]
    torch::Tensor layer_D,               // [n_layers, H]
    torch::Tensor layer_dt_bias,          // [n_layers, H]
    torch::Tensor layer_ssm_norm_weight,  // [n_layers, d_inner]
    torch::Tensor layer_out_proj_int8,    // [n_layers, hidden, d_inner] INT8
    torch::Tensor layer_out_proj_scale,   // [n_layers, hidden] FP32
    torch::Tensor layer_out_proj_row_sum, // [n_layers, hidden] INT32
    torch::Tensor layer_out_proj_bias,    // [n_layers, hidden] FP32
    // Cache state (modified in-place)
    torch::Tensor conv_states,    // [n_layers, K, conv_dim]
    torch::Tensor ssm_states,     // [n_layers, H, D, N]
    // Architecture dims
    int64_t n_layers,
    int64_t hidden_size,
    int64_t d_inner,
    int64_t conv_dim,
    int64_t n_heads,
    int64_t head_dim,
    int64_t state_size,
    int64_t n_groups,
    int64_t conv_kernel,
    float time_step_lo,
    float time_step_hi,
    int64_t d_mlp,
    int64_t proj_size,
    // Optional guidance deltas — FP32 [n_layers, d_inner] or empty
    torch::Tensor guidance_deltas
) {
    const int H = n_heads;
    const int D = head_dim;
    const int N = state_size;
    const int K = conv_kernel;
    const int vocab_size = embed_weight_int8.size(0);

    // Working buffers
    std::vector<float> hidden(hidden_size);
    std::vector<float> normed(hidden_size);
    std::vector<float> projected(proj_size);
    std::vector<float> conv_out(conv_dim);
    std::vector<float> ssm_y(H * D);
    std::vector<float> ssm_h_new(H * D * N);
    std::vector<float> gated_out(d_inner);
    std::vector<float> layer_out(hidden_size);
    // Quantization buffers
    std::vector<uint8_t> x_u8_hidden(hidden_size);
    std::vector<uint8_t> x_u8_inner(d_inner);

    // 1. Embedding lookup: INT8 → dequant to FP32
    const int8_t* embed_ptr = embed_weight_int8.data_ptr<int8_t>();
    const float* embed_scale_ptr = embed_scale.data_ptr<float>();
    {
        const int8_t* row = embed_ptr + (int64_t)token_id * hidden_size;
        float scale = embed_scale_ptr[token_id];
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] = (float)row[i] * scale;
        }
    }

    // Pointers
    float* norm_f_ptr = norm_f_weight.data_ptr<float>();
    float* ln_w_ptr = layer_norm_weights.data_ptr<float>();
    const int8_t* in_proj_ptr = layer_in_proj_int8.data_ptr<int8_t>();
    const float* in_proj_scale_ptr = layer_in_proj_scale.data_ptr<float>();
    const int32_t* in_proj_rsum_ptr = layer_in_proj_row_sum.data_ptr<int32_t>();
    float* in_proj_bias_ptr = layer_in_proj_bias.data_ptr<float>();
    float* conv_w_ptr = layer_conv_weight.data_ptr<float>();
    float* conv_b_ptr = layer_conv_bias.data_ptr<float>();
    float* alog_ptr = layer_A_log.data_ptr<float>();
    float* d_ptr = layer_D.data_ptr<float>();
    float* dtb_ptr = layer_dt_bias.data_ptr<float>();
    float* ssm_nw_ptr = layer_ssm_norm_weight.data_ptr<float>();
    const int8_t* out_proj_ptr = layer_out_proj_int8.data_ptr<int8_t>();
    const float* out_proj_scale_ptr = layer_out_proj_scale.data_ptr<float>();
    const int32_t* out_proj_rsum_ptr = layer_out_proj_row_sum.data_ptr<int32_t>();
    float* out_proj_bias_ptr = layer_out_proj_bias.data_ptr<float>();
    float* cs_ptr = conv_states.data_ptr<float>();
    float* ss_ptr = ssm_states.data_ptr<float>();

    // Strides
    const int64_t in_proj_layer_stride = (int64_t)proj_size * hidden_size;
    const int64_t out_proj_layer_stride = (int64_t)hidden_size * d_inner;
    const int64_t conv_w_layer_stride = (int64_t)conv_dim * K;
    const int64_t cs_layer_stride = (int64_t)conv_dim * K;
    const int64_t ss_layer_stride = (int64_t)H * D * N;

    // Guidance delta pointer (may be null if no guidance)
    const float* guide_ptr = (guidance_deltas.defined() && guidance_deltas.numel() > 0)
        ? guidance_deltas.data_ptr<float>() : nullptr;

    // 2. Run through all layers
    for (int layer = 0; layer < n_layers; layer++) {
        // RMSNorm
        rms_norm(normed.data(), hidden.data(),
                 ln_w_ptr + layer * hidden_size, hidden_size);

        // in_proj (INT8 GEMV): [proj_size, hidden] @ [hidden] → [proj_size]
        float x_scale = quantize_input_to_uint8(
            x_u8_hidden.data(), normed.data(), hidden_size);
        gemv_int8(projected.data(), normed.data(),
                  in_proj_ptr + layer * in_proj_layer_stride,
                  in_proj_scale_ptr + layer * proj_size,
                  in_proj_rsum_ptr + layer * proj_size,
                  in_proj_bias_ptr + layer * proj_size,
                  proj_size, hidden_size,
                  x_scale, x_u8_hidden.data());

        // Split: [d_mlp, d_mlp, gate(d_inner), hbc(conv_dim), dt(H)]
        float* gate = projected.data() + 2 * d_mlp;
        float* hbc = gate + d_inner;
        float* dt = hbc + conv_dim;

        // Inject guidance delta into x-branch (first d_inner of hbc)
        if (guide_ptr) {
            const float* layer_delta = guide_ptr + layer * d_inner;
            for (int i = 0; i < d_inner; i++) {
                hbc[i] += layer_delta[i];
            }
        }

        // Conv1d cached step + SiLU
        conv_silu_step(conv_out.data(),
                       cs_ptr + layer * cs_layer_stride,
                       hbc, conv_w_ptr + layer * conv_w_layer_stride,
                       conv_b_ptr + layer * conv_dim,
                       conv_dim, K);

        // Split conv output: x(d_inner) + B(n_groups*N) + C(n_groups*N)
        float* x_ssm = conv_out.data();
        float* B_raw = x_ssm + d_inner;
        float* C_raw = B_raw + n_groups * N;

        // Reshape B, C: [n_groups, N] → [H, N]
        std::vector<float> B_expanded(H * N);
        std::vector<float> C_expanded(H * N);
        int heads_per_group = H / n_groups;
        for (int g = 0; g < n_groups; g++) {
            for (int hg = 0; hg < heads_per_group; hg++) {
                int h_idx = g * heads_per_group + hg;
                std::memcpy(B_expanded.data() + h_idx * N, B_raw + g * N, N * sizeof(float));
                std::memcpy(C_expanded.data() + h_idx * N, C_raw + g * N, N * sizeof(float));
            }
        }

        // dt expansion
        std::vector<float> dt_expanded(H * D);
        for (int h = 0; h < H; h++) {
            for (int d = 0; d < D; d++) {
                dt_expanded[h * D + d] = dt[h];
            }
        }

        // A = -exp(A_log)
        std::vector<float> A_neg(H);
        for (int h = 0; h < H; h++) {
            A_neg[h] = -std::exp(alog_ptr[layer * H + h]);
        }

        // SSM step
        ssm_step(ssm_y.data(), ssm_h_new.data(),
                 x_ssm, B_expanded.data(), C_expanded.data(),
                 dt_expanded.data(), A_neg.data(),
                 d_ptr + layer * H,
                 ss_ptr + layer * ss_layer_stride,
                 dtb_ptr + layer * H,
                 time_step_lo, time_step_hi,
                 H, D, N);

        // Copy new state back
        std::memcpy(ss_ptr + layer * ss_layer_stride, ssm_h_new.data(),
                     H * D * N * sizeof(float));

        // Gated RMSNorm
        rms_norm_gated(gated_out.data(), ssm_y.data(), gate,
                       ssm_nw_ptr + layer * d_inner, d_inner);

        // out_proj (INT8 GEMV): [hidden, d_inner] @ [d_inner] → [hidden]
        float x_scale_out = quantize_input_to_uint8(
            x_u8_inner.data(), gated_out.data(), d_inner);
        gemv_int8(layer_out.data(), gated_out.data(),
                  out_proj_ptr + layer * out_proj_layer_stride,
                  out_proj_scale_ptr + layer * hidden_size,
                  out_proj_rsum_ptr + layer * hidden_size,
                  out_proj_bias_ptr + layer * hidden_size,
                  hidden_size, d_inner,
                  x_scale_out, x_u8_inner.data());

        // Residual add
        for (int i = 0; i < hidden_size; i++) {
            hidden[i] += layer_out[i];
        }
    }

    // 3. Final RMSNorm
    rms_norm(normed.data(), hidden.data(), norm_f_ptr, hidden_size);

    // 4. LM head (INT8 GEMV): [vocab_size, hidden] @ [hidden] → [vocab_size]
    float x_scale_lm = quantize_input_to_uint8(
        x_u8_hidden.data(), normed.data(), hidden_size);
    auto logits = torch::empty({1, vocab_size}, torch::kFloat32);
    gemv_int8(logits.data_ptr<float>(), normed.data(),
              embed_ptr, embed_scale_ptr,
              embed_row_sum.data_ptr<int32_t>(),
              nullptr, vocab_size, hidden_size,
              x_scale_lm, x_u8_hidden.data());

    return logits;
}


PYBIND11_MODULE(_fused_mamba2_forward, m) {
    m.doc() = "Fused Mamba2-65M single-step forward with BF16/INT8 weights";
    m.def("fused_mamba2_step", &fused_mamba2_step,
          "Fused full model single-step forward (BF16 weights)");
    m.def("fused_mamba2_step_int8", &fused_mamba2_step_int8,
          "Fused full model single-step forward (INT8 weights, ~2x less memory)");
}
