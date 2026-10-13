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
            *loss -= std::log(std::max(probabilities[row * cols + target], 1.0e-30f));
        }
        *loss /= static_cast<float>(rows);
    }

    void cross_entropy_backward(const float* probabilities, const float* targets,
                                const float* upstream, float* grad,
                                int rows, int cols) override {
        const float scale = upstream[0] / static_cast<float>(rows);
        for (int row = 0; row < rows; ++row) {
            const int target = static_cast<int>(targets[row]);
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
