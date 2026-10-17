#include "tinydl/backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace tinydl {
namespace {

class CpuBackend final : public Backend {
public:
    DeviceType type() const noexcept override { return DeviceType::CPU; }

    void* allocate(std::size_t bytes) override {
        if (void* ptr = std::malloc(bytes)) return ptr;
        throw std::bad_alloc();
    }

    void deallocate(void* ptr) noexcept override { std::free(ptr); }

    void copy_from_host(void* dst, const float* src, std::size_t count) override {
        std::copy(src, src + count, static_cast<float*>(dst));
    }

    void copy_to_host(float* dst, const void* src, std::size_t count) override {
        const auto* input = static_cast<const float*>(src);
        std::copy(input, input + count, dst);
    }

    void copy(float* dst, const float* src, std::size_t count) override {
        std::copy(src, src + count, dst);
    }

    void synchronize() override {}

    void fill(float* out, float value, std::size_t n) override {
        std::fill(out, out + n, value);
    }

    void add(const float* a, const float* b, float* out, std::size_t n) override {
        for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
    }

    void gelu(const float* x, float* out, std::size_t n) override {
        constexpr float k = 0.7978845608028654f;
        for (std::size_t i = 0; i < n; ++i) {
            const float v = x[i];
            out[i] = 0.5f * v * (1.0f + std::tanh(k * (v + 0.044715f * v * v * v)));
        }
    }

    void matmul(const float* a, const float* b, float* out,
                int m, int n, int k) override {
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                float sum = 0.0f;
                for (int inner = 0; inner < k; ++inner) {
                    sum += a[row * k + inner] * b[inner * n + col];
                }
                out[row * n + col] = sum;
            }
        }
    }

    void matmul_backward_left(const float* grad, const float* b, float* out,
                              int m, int n, int k) override {
        for (int row = 0; row < m; ++row)
            for (int inner = 0; inner < k; ++inner) {
                float sum = 0.0f;
                for (int col = 0; col < n; ++col)
                    sum += grad[row * n + col] * b[inner * n + col];
                out[row * k + inner] = sum;
            }
    }

    void matmul_backward_right(const float* a, const float* grad, float* out,
                               int m, int n, int k) override {
        for (int inner = 0; inner < k; ++inner)
            for (int col = 0; col < n; ++col) {
                float sum = 0.0f;
                for (int row = 0; row < m; ++row)
                    sum += a[row * k + inner] * grad[row * n + col];
                out[inner * n + col] = sum;
            }
    }

    void softmax_rows(const float* x, float* out, int rows, int cols) override {
        for (int row = 0; row < rows; ++row) {
            const float* input = x + row * cols;
            float* output = out + row * cols;
            float maximum = input[0];
            for (int col = 1; col < cols; ++col) maximum = std::max(maximum, input[col]);
            float denominator = 0.0f;
            for (int col = 0; col < cols; ++col) {
                output[col] = std::exp(input[col] - maximum);
                denominator += output[col];
            }
            for (int col = 0; col < cols; ++col) output[col] /= denominator;
        }
    }

    void layer_norm(const float* x, const float* weight, const float* bias,
                    float* out, int rows, int cols, float eps) override {
        for (int row = 0; row < rows; ++row) {
            const float* input = x + row * cols;
            float mean = 0.0f;
            for (int col = 0; col < cols; ++col) mean += input[col];
            mean /= static_cast<float>(cols);
            float variance = 0.0f;
            for (int col = 0; col < cols; ++col) {
                const float delta = input[col] - mean;
                variance += delta * delta;
            }
            variance /= static_cast<float>(cols);
            const float inv_std = 1.0f / std::sqrt(variance + eps);
            for (int col = 0; col < cols; ++col) {
                out[row * cols + col] = (input[col] - mean) * inv_std * weight[col] + bias[col];
            }
        }
    }

    void causal_attention_forward(const float* q, const float* k, const float* v,
                                  float* probabilities, float* out,
                                  int tokens, int channels, int heads) override {
        const int head_dim = channels / heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::fill(probabilities,
                  probabilities + static_cast<std::size_t>(heads) * tokens * tokens, 0.0f);
        std::fill(out, out + static_cast<std::size_t>(tokens) * channels, 0.0f);
        for (int head = 0; head < heads; ++head) {
            const int head_offset = head * head_dim;
            for (int query = 0; query < tokens; ++query) {
                float maximum = -INFINITY;
                for (int key = 0; key <= query; ++key) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        score += q[query * channels + head_offset + d] *
                                 k[key * channels + head_offset + d];
                    score *= scale;
                    probabilities[(head * tokens + query) * tokens + key] = score;
                    maximum = std::max(maximum, score);
                }
                float denominator = 0.0f;
                for (int key = 0; key <= query; ++key) {
                    float& probability = probabilities[(head * tokens + query) * tokens + key];
                    probability = std::exp(probability - maximum);
                    denominator += probability;
                }
                for (int key = 0; key <= query; ++key) {
                    const float probability =
                        probabilities[(head * tokens + query) * tokens + key] / denominator;
                    probabilities[(head * tokens + query) * tokens + key] = probability;
                    for (int d = 0; d < head_dim; ++d)
                        out[query * channels + head_offset + d] +=
                            probability * v[key * channels + head_offset + d];
                }
            }
        }
    }

    void causal_attention_backward(const float* q, const float* k, const float* v,
                                   const float* probabilities, const float* grad,
                                   float* dq, float* dk, float* dv,
                                   int tokens, int channels, int heads) override {
        const int head_dim = channels / heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const std::size_t elements = static_cast<std::size_t>(tokens) * channels;
        std::fill(dq, dq + elements, 0.0f);
        std::fill(dk, dk + elements, 0.0f);
        std::fill(dv, dv + elements, 0.0f);
        for (int head = 0; head < heads; ++head) {
            const int head_offset = head * head_dim;
            for (int query = 0; query < tokens; ++query) {
                float softmax_dot = 0.0f;
                for (int key = 0; key <= query; ++key) {
                    float dp = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        dp += grad[query * channels + head_offset + d] *
                              v[key * channels + head_offset + d];
                    softmax_dot += probabilities[(head * tokens + query) * tokens + key] * dp;
                }
                for (int key = 0; key <= query; ++key) {
                    float dp = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                        dp += grad[query * channels + head_offset + d] *
                              v[key * channels + head_offset + d];
                    const float probability =
                        probabilities[(head * tokens + query) * tokens + key];
                    const float ds = probability * (dp - softmax_dot) * scale;
                    for (int d = 0; d < head_dim; ++d) {
                        const int query_index = query * channels + head_offset + d;
                        const int key_index = key * channels + head_offset + d;
                        dq[query_index] += ds * k[key_index];
                        dk[key_index] += ds * q[query_index];
                        dv[key_index] += probability * grad[query_index];
                    }
                }
            }
        }
    }


    float squared_l2_norm(const float* x, std::size_t n) override {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) sum += static_cast<double>(x[i]) * x[i];
        return static_cast<float>(sum);
    }

    void adamw_update(float* values, const float* gradients,
                      float* first_moment, float* second_moment, std::size_t n,
                      float beta1, float beta2, float correction1, float correction2,
                      float learning_rate, float epsilon, float weight_decay,
                      float gradient_scale) override {
        for (std::size_t i = 0; i < n; ++i) {
            const float gradient = gradients[i] * gradient_scale;
            first_moment[i] = beta1 * first_moment[i] + (1.0f - beta1) * gradient;
            second_moment[i] = beta2 * second_moment[i] + (1.0f - beta2) * gradient * gradient;
            const float first = first_moment[i] / correction1;
            const float second = second_moment[i] / correction2;
            values[i] -= learning_rate *
                (first / (std::sqrt(second) + epsilon) + weight_decay * values[i]);
        }
    }

    void cross_entropy_forward(const float* logits, const float* targets,
                               const float* weights, float normalizer,
                               float* probabilities, float* loss, int rows, int cols) override {
        *loss = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const float* input = logits + row * cols;
            float maximum = input[0];
            for (int col = 1; col < cols; ++col) maximum = std::max(maximum, input[col]);
            float denominator = 0.0f;
            for (int col = 0; col < cols; ++col) {
                probabilities[row * cols + col] = std::exp(input[col] - maximum);
                denominator += probabilities[row * cols + col];
            }
            for (int col = 0; col < cols; ++col) probabilities[row * cols + col] /= denominator;
            const int target = static_cast<int>(targets[row]);
            *loss -= weights[row] *
                std::log(std::max(probabilities[row * cols + target], 1.0e-30f));
        }
        *loss /= normalizer;
    }

    void cross_entropy_backward(const float* probabilities, const float* targets,
                                const float* weights, float normalizer,
                                const float* upstream, float* grad,
                                int rows, int cols) override {
        for (int row = 0; row < rows; ++row) {
            const int target = static_cast<int>(targets[row]);
            const float scale = upstream[0] * weights[row] / normalizer;
            for (int col = 0; col < cols; ++col) {
                const float delta = probabilities[row * cols + col] - (col == target ? 1.0f : 0.0f);
                grad[row * cols + col] = delta * scale;
            }
        }
    }
};

}  // namespace

std::shared_ptr<Backend> make_cpu_backend() { return std::make_shared<CpuBackend>(); }

}  // namespace tinydl
