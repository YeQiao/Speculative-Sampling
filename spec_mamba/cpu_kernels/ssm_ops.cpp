/*
 * CPU-optimized SSM operations with AVX-512 intrinsics.
 *
 * Core operation: single-step SSM state update
 *   h_new[b,h,d,n] = h_old[b,h,d,n] * exp(dt[b,h,d] * A[h]) + dt[b,h,d] * B[b,h,n] * x[b,h,d]
 *   y[b,h,d] = sum_n(C[b,h,n] * h_new[b,h,d,n]) + D[h] * x[b,h,d]
 *
 * Shapes (Mamba2-65M):
 *   B=1, H=16 heads, D=64 head_dim, N=128 state_size
 *   h: [B, H, D, N] = [1, 16, 64, 128]
 *
 * The inner loop over N=128 maps perfectly to 8 AVX-512 registers (128 / 16 = 8).
 */

#include <torch/extension.h>
#include <cmath>
#include <tuple>

#ifdef __AVX512F__
#include <immintrin.h>
#define HAS_AVX512 1
#else
#define HAS_AVX512 0
#endif

// Softplus: log(1 + exp(x))
inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return std::log(1.0f + std::exp(x));
}

/*
 * SSM single-step update with AVX-512.
 *
 * For each (batch, head, head_dim) position:
 *   dt_val = clamp(softplus(dt + dt_bias), lo, hi)
 *   dA = exp(dt_val * A)                          -- scalar
 *   dBx = dt_val * B[n] * x                       -- vector over n
 *   h_new[n] = h_old[n] * dA + dBx[n]             -- vector over n
 *   y += C[n] * h_new[n]                           -- reduction over n
 */
std::tuple<torch::Tensor, torch::Tensor> ssm_step_avx512(
    torch::Tensor x,           // [B, H, D]
    torch::Tensor B_ssm,       // [B, H, N]
    torch::Tensor C_ssm,       // [B, H, N]
    torch::Tensor dt,          // [B, H, D]
    torch::Tensor A,           // [H]
    torch::Tensor D_skip,      // [H]
    torch::Tensor ssm_state,   // [B, H, D, N]
    torch::Tensor dt_bias,     // [H]
    float time_step_lo,
    float time_step_hi
) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(ssm_state.is_contiguous() && ssm_state.scalar_type() == torch::kFloat32);

    const int B = x.size(0);
    const int H = x.size(1);
    const int head_D = x.size(2);
    const int N = ssm_state.size(3);

    auto new_state = torch::empty_like(ssm_state);
    auto y = torch::zeros({B, H, head_D}, torch::kFloat32);

    float* x_ptr = x.data_ptr<float>();
    float* B_ptr = B_ssm.data_ptr<float>();
    float* C_ptr = C_ssm.data_ptr<float>();
    float* dt_ptr = dt.data_ptr<float>();
    float* A_ptr = A.data_ptr<float>();
    float* D_ptr = D_skip.data_ptr<float>();
    float* h_old_ptr = ssm_state.data_ptr<float>();
    float* h_new_ptr = new_state.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    float* dt_bias_ptr = dt_bias.data_ptr<float>();

    #pragma omp parallel for collapse(3) schedule(static)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int d = 0; d < head_D; d++) {
                // Compute indices
                const int bh = b * H + h;
                const int bhd = bh * head_D + d;

                // dt with bias and activation
                float dt_val = dt_ptr[bhd] + dt_bias_ptr[h];
                dt_val = softplus(dt_val);
                if (dt_val < time_step_lo) dt_val = time_step_lo;
                if (dt_val > time_step_hi) dt_val = time_step_hi;

                // dA = exp(dt * A[h]) -- A is negative
                float A_val = A_ptr[h];
                float dA_val = std::exp(dt_val * A_val);

                // x value and dBx scale
                float x_val = x_ptr[bhd];
                float dBx_scale = dt_val * x_val;

                // Pointer to B[b,h,:] and C[b,h,:] and h_old[b,h,d,:] and h_new[b,h,d,:]
                const float* B_bh = B_ptr + bh * N;
                const float* C_bh = C_ptr + bh * N;
                const float* h_old = h_old_ptr + (bhd) * N;
                float* h_new = h_new_ptr + (bhd) * N;

                float y_accum = 0.0f;

#if HAS_AVX512
                // Process N elements using AVX-512 (16 floats per register)
                __m512 v_dA = _mm512_set1_ps(dA_val);
                __m512 v_dBx_scale = _mm512_set1_ps(dBx_scale);
                __m512 v_y_accum = _mm512_setzero_ps();

                int n = 0;
                for (; n + 16 <= N; n += 16) {
                    // Load h_old[n:n+16]
                    __m512 v_h_old = _mm512_loadu_ps(h_old + n);
                    // Load B[n:n+16]
                    __m512 v_B = _mm512_loadu_ps(B_bh + n);
                    // Load C[n:n+16]
                    __m512 v_C = _mm512_loadu_ps(C_bh + n);

                    // h_new = h_old * dA + dBx_scale * B
                    __m512 v_h_new = _mm512_fmadd_ps(v_h_old, v_dA,
                                        _mm512_mul_ps(v_dBx_scale, v_B));

                    // Store h_new
                    _mm512_storeu_ps(h_new + n, v_h_new);

                    // y_accum += C * h_new
                    v_y_accum = _mm512_fmadd_ps(v_C, v_h_new, v_y_accum);
                }

                y_accum = _mm512_reduce_add_ps(v_y_accum);

                // Handle remaining elements
                for (; n < N; n++) {
                    float h_n = h_old[n] * dA_val + dBx_scale * B_bh[n];
                    h_new[n] = h_n;
                    y_accum += C_bh[n] * h_n;
                }
#else
                // Scalar fallback
                for (int n = 0; n < N; n++) {
                    float h_n = h_old[n] * dA_val + dBx_scale * B_bh[n];
                    h_new[n] = h_n;
                    y_accum += C_bh[n] * h_n;
                }
#endif

                // D skip connection: y += D[h] * x
                y_accum += D_ptr[h] * x_val;

                y_ptr[bhd] = y_accum;
            }
        }
    }

    return std::make_tuple(y, new_state);
}


/*
 * Fused conv1d + SiLU for single-step cached inference.
 *
 * conv_states: [B, conv_dim, conv_kernel]
 * new_input: [B, conv_dim]
 * conv_weight: [conv_dim, conv_kernel]
 * conv_bias: [conv_dim] or empty
 *
 * 1. Shift conv_states left, append new_input
 * 2. out = sum(conv_states * conv_weight, dim=-1) + bias
 * 3. out = silu(out)
 */
torch::Tensor fused_conv_silu_step(
    torch::Tensor conv_states,  // [B, conv_dim, conv_kernel] -- modified in place
    torch::Tensor new_input,    // [B, conv_dim]
    torch::Tensor conv_weight,  // [conv_dim, conv_kernel]
    torch::Tensor conv_bias     // [conv_dim] or empty
) {
    const int B = conv_states.size(0);
    const int conv_dim = conv_states.size(1);
    const int K = conv_states.size(2);

    float* cs_ptr = conv_states.data_ptr<float>();
    float* in_ptr = new_input.data_ptr<float>();
    float* w_ptr = conv_weight.data_ptr<float>();

    bool has_bias = conv_bias.numel() > 0;
    float* b_ptr = has_bias ? conv_bias.data_ptr<float>() : nullptr;

    auto output = torch::empty({B, conv_dim}, torch::kFloat32);
    float* out_ptr = output.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < conv_dim; c++) {
            float* cs = cs_ptr + (b * conv_dim + c) * K;

            // Shift left
            for (int k = 0; k < K - 1; k++) {
                cs[k] = cs[k + 1];
            }
            cs[K - 1] = in_ptr[b * conv_dim + c];

            // Dot product with conv_weight
            float val = 0.0f;
            const float* w = w_ptr + c * K;
            for (int k = 0; k < K; k++) {
                val += cs[k] * w[k];
            }
            if (has_bias) {
                val += b_ptr[c];
            }
            // SiLU
            out_ptr[b * conv_dim + c] = val / (1.0f + std::exp(-val));
        }
    }

    return output;
}


PYBIND11_MODULE(_cpu_ssm_ops, m) {
    m.doc() = "CPU-optimized SSM operations with AVX-512";
    m.def("ssm_step", &ssm_step_avx512,
          "SSM single-step state update with AVX-512 intrinsics",
          py::arg("x"), py::arg("B_ssm"), py::arg("C_ssm"),
          py::arg("dt"), py::arg("A"), py::arg("D"),
          py::arg("ssm_state"), py::arg("dt_bias"),
          py::arg("time_step_lo"), py::arg("time_step_hi"));
    m.def("fused_conv_silu_step", &fused_conv_silu_step,
          "Fused conv1d + SiLU for cached single-step inference",
          py::arg("conv_states"), py::arg("new_input"),
          py::arg("conv_weight"), py::arg("conv_bias"));
}
