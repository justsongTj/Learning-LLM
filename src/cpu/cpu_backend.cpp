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
};

}  // namespace

std::shared_ptr<Backend> make_cpu_backend() { return std::make_shared<CpuBackend>(); }

}  // namespace tinydl
