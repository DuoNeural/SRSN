#include <torch/extension.h>
#include <cmath>
#include <omp.h>
#include <vector>

// Fast inline scalar helpers for SIMD vectorization
inline float fast_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

inline float fast_dsilu(float x) {
    float sig = 1.0f / (1.0f + std::exp(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

// Forward Pass: Returns {outputs, q_hist, p_hist}
// Shapes: F_ext, gamma: [B, T, D] (contiguous float32)
//         omega_sq, alpha: [D] (contiguous float32)
std::vector<torch::Tensor> sdoc_forward_cpu(
    torch::Tensor F_ext,
    torch::Tensor gamma,
    torch::Tensor omega_sq,
    torch::Tensor alpha,
    double dt_val
) {
    auto F_c = F_ext.contiguous();
    auto G_c = gamma.contiguous();
    auto W_c = omega_sq.contiguous();
    auto A_c = alpha.contiguous();

    const int64_t B = F_c.size(0);
    const int64_t T = F_c.size(1);
    const int64_t D = F_c.size(2);
    const float dt = static_cast<float>(dt_val);

    auto outputs = torch::empty_like(F_c);
    auto q_hist = torch::empty_like(F_c);
    auto p_hist = torch::empty_like(F_c);

    const float* F_ptr = F_c.data_ptr<float>();
    const float* G_ptr = G_c.data_ptr<float>();
    const float* W_ptr = W_c.data_ptr<float>();
    const float* A_ptr = A_c.data_ptr<float>();

    float* Y_ptr = outputs.data_ptr<float>();
    float* Q_ptr = q_hist.data_ptr<float>();
    float* P_ptr = p_hist.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        std::vector<float> q(D, 0.0f);
        std::vector<float> p(D, 0.0f);

        for (int64_t t = 0; t < T; ++t) {
            const int64_t offset = (b * T + t) * D;
            #pragma omp simd
            for (int64_t d = 0; d < D; ++d) {
                float f_t = F_ptr[offset + d];
                float g_t = G_ptr[offset + d];
                float w_d = W_ptr[d];
                float a_d = A_ptr[d];

                float q_prev = q[d];
                float p_prev = p[d];
                float th_q = std::tanh(q_prev);

                float p_new = (1.0f - dt * g_t) * p_prev + dt * (f_t - w_d * q_prev - a_d * th_q);
                p_new = std::fmax(-50.0f, std::fmin(50.0f, p_new));
                float q_new = q_prev + dt * p_new;
                q_new = std::fmax(-50.0f, std::fmin(50.0f, q_new));

                p[d] = p_new;
                q[d] = q_new;

                P_ptr[offset + d] = p_new;
                Q_ptr[offset + d] = q_new;
                Y_ptr[offset + d] = q_new * fast_silu(p_new);
            }
        }
    }
    return {outputs, q_hist, p_hist};
}

// Analytical Adjoint Backward Pass: Returns {dF_ext, dgamma, domega_sq, dalpha}
std::vector<torch::Tensor> sdoc_backward_cpu(
    torch::Tensor grad_out,
    torch::Tensor gamma,
    torch::Tensor q_hist,
    torch::Tensor p_hist,
    torch::Tensor omega_sq,
    torch::Tensor alpha,
    double dt_val
) {
    auto dY_c = grad_out.contiguous();
    auto G_c = gamma.contiguous();
    auto Q_c = q_hist.contiguous();
    auto P_c = p_hist.contiguous();
    auto W_c = omega_sq.contiguous();
    auto A_c = alpha.contiguous();

    const int64_t B = dY_c.size(0);
    const int64_t T = dY_c.size(1);
    const int64_t D = dY_c.size(2);
    const float dt = static_cast<float>(dt_val);

    auto dF_ext = torch::empty_like(dY_c);
    auto dgamma = torch::empty_like(dY_c);
    auto domega_sq_batch = torch::zeros({B, D}, dY_c.options());
    auto dalpha_batch = torch::zeros({B, D}, dY_c.options());

    const float* dY_ptr = dY_c.data_ptr<float>();
    const float* G_ptr = G_c.data_ptr<float>();
    const float* Q_ptr = Q_c.data_ptr<float>();
    const float* P_ptr = P_c.data_ptr<float>();
    const float* W_ptr = W_c.data_ptr<float>();
    const float* A_ptr = A_c.data_ptr<float>();

    float* dF_ptr = dF_ext.data_ptr<float>();
    float* dG_ptr = dgamma.data_ptr<float>();
    float* dW_b_ptr = domega_sq_batch.data_ptr<float>();
    float* dA_b_ptr = dalpha_batch.data_ptr<float>();

    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < B; ++b) {
        std::vector<float> dq_next(D, 0.0f);
        std::vector<float> dp_next(D, 0.0f);
        float* dW_local = dW_b_ptr + b * D;
        float* dA_local = dA_b_ptr + b * D;

        for (int64_t t = T - 1; t >= 0; --t) {
            const int64_t offset = (b * T + t) * D;
            const int64_t prev_offset = (t > 0) ? (b * T + (t - 1)) * D : -1;

            #pragma omp simd
            for (int64_t d = 0; d < D; ++d) {
                float dy = dY_ptr[offset + d];
                float q_t = Q_ptr[offset + d];
                float p_t = P_ptr[offset + d];
                float g_t = G_ptr[offset + d];
                float w_d = W_ptr[d];
                float a_d = A_ptr[d];

                float q_prev = (t > 0) ? Q_ptr[prev_offset + d] : 0.0f;
                float p_prev = (t > 0) ? P_ptr[prev_offset + d] : 0.0f;

                float silu_p = fast_silu(p_t);
                float dsilu_p = fast_dsilu(p_t);

                float dq = dy * silu_p + dq_next[d];
                dq = std::fmax(-50.0f, std::fmin(50.0f, dq));
                float dp = dy * q_t * dsilu_p + dp_next[d] + dt * dq;
                dp = std::fmax(-50.0f, std::fmin(50.0f, dp));

                dF_ptr[offset + d] = dt * dp;
                dG_ptr[offset + d] = -dt * p_prev * dp;

                float th_q_prev = std::tanh(q_prev);
                float sech2_q_prev = 1.0f - th_q_prev * th_q_prev;

                dW_local[d] += -dt * q_prev * dp;
                dA_local[d] += -dt * th_q_prev * dp;

                float dq_val = dq - dt * dp * (w_d + a_d * sech2_q_prev);
                dq_next[d] = std::fmax(-50.0f, std::fmin(50.0f, dq_val));
                float dp_val = (1.0f - dt * g_t) * dp;
                dp_next[d] = std::fmax(-50.0f, std::fmin(50.0f, dp_val));
            }
        }
    }

    auto domega_sq = domega_sq_batch.sum(0);
    auto dalpha = dalpha_batch.sum(0);
    return {dF_ext, dgamma, domega_sq, dalpha};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sdoc_forward_cpu, "SDOC Forward CPU (OpenMP + AVX2)");
    m.def("backward", &sdoc_backward_cpu, "SDOC Backward CPU (OpenMP + AVX2)");
}
