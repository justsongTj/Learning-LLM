#include "tinydl/backend.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace tinydl {
namespace {

void check_hip(hipError_t status, const char* operation) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

void check_blas(hipblasStatus_t status, const char* operation) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with hipBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

__global__ void fill_kernel(float* out, float value, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

__global__ void add_kernel(const float* a, const float* b, float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void gelu_kernel(const float* x, float* out, std::size_t n) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        constexpr float k = 0.7978845608028654f;
        const float v = x[i];
        out[i] = 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
    }
}

__global__ void matmul_backward_left_kernel(const float* grad, const float* b, float* out,
                                            int m, int n, int k) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(m) * k;
    if (index >= count) return;
    const int row = static_cast<int>(index / k);
    const int inner = static_cast<int>(index % k);
    float sum = 0.0f;
    for (int col = 0; col < n; ++col)
        sum += grad[row * n + col] * b[inner * n + col];
    out[index] = sum;
}

__global__ void matmul_backward_right_kernel(const float* a, const float* grad, float* out,
                                             int m, int n, int k) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(k) * n;
    if (index >= count) return;
    const int inner = static_cast<int>(index / n);
    const int col = static_cast<int>(index % n);
    float sum = 0.0f;
    for (int row = 0; row < m; ++row)
        sum += a[row * k + inner] * grad[row * n + col];
    out[index] = sum;
}

__global__ void squared_l2_norm_kernel(const float* x, float* result, std::size_t n) {
    __shared__ float partial[256];
    const unsigned tid = threadIdx.x;
    float sum = 0.0f;
    for (std::size_t i = blockIdx.x * blockDim.x + tid;
         i < n; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        sum += x[i] * x[i];
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, partial[0]);
}

__global__ void adamw_kernel(float* values, const float* gradients,
                             float* first_moment, float* second_moment, std::size_t n,
                             float beta1, float beta2, float correction1, float correction2,
                             float learning_rate, float epsilon, float weight_decay,
                             float gradient_scale) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gradient = gradients[i] * gradient_scale;
    const float first_value = beta1 * first_moment[i] + (1.0f - beta1) * gradient;
    const float second_value = beta2 * second_moment[i] +
                               (1.0f - beta2) * gradient * gradient;
    first_moment[i] = first_value;
    second_moment[i] = second_value;
    const float first = first_value / correction1;
    const float second = second_value / correction2;
    values[i] -= learning_rate *
        (first / (sqrtf(second) + epsilon) + weight_decay * values[i]);
}

__global__ void cross_entropy_forward_kernel(const float* logits, const float* targets,
                                             float* probabilities, float* loss,
                                             int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    const float* input = logits + row * cols;
    float* output = probabilities + row * cols;
    float maximum = input[0];
    for (int col = 1; col < cols; ++col) maximum = fmaxf(maximum, input[col]);
    float denominator = 0.0f;
    for (int col = 0; col < cols; ++col) {
        output[col] = expf(input[col] - maximum);
        denominator += output[col];
    }
    for (int col = 0; col < cols; ++col) output[col] /= denominator;
    const int target = static_cast<int>(targets[row]);
    atomicAdd(loss, -logf(fmaxf(output[target], 1.0e-30f)) / static_cast<float>(rows));
}

__global__ void cross_entropy_backward_kernel(const float* probabilities, const float* targets,
                                              const float* upstream, float* grad,
                                              int rows, int cols) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    if (index >= count) return;
    const int row = static_cast<int>(index / cols);
    const int col = static_cast<int>(index % cols);
    const int target = static_cast<int>(targets[row]);
    const float delta = probabilities[index] - (col == target ? 1.0f : 0.0f);
    grad[index] = delta * upstream[0] / static_cast<float>(rows);
}

__global__ void softmax_rows_kernel(const float* x, float* out, int cols) {
    const int row = blockIdx.x;
    if (threadIdx.x != 0) return;
    const float* input = x + row * cols;
    float* output = out + row * cols;
    float maximum = input[0];
    for (int col = 1; col < cols; ++col) maximum = fmaxf(maximum, input[col]);
    float denominator = 0.0f;
    for (int col = 0; col < cols; ++col) {
        output[col] = expf(input[col] - maximum);
        denominator += output[col];
    }
    for (int col = 0; col < cols; ++col) output[col] /= denominator;
}

__global__ void layer_norm_kernel(const float* x, const float* weight, const float* bias,
                                  float* out, int cols, float eps) {
    const int row = blockIdx.x;
    if (threadIdx.x != 0) return;
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
    const float inv_std = rsqrtf(variance + eps);
    for (int col = 0; col < cols; ++col) {
        out[row * cols + col] = (input[col] - mean) * inv_std * weight[col] + bias[col];
    }
}

class DcuBackend final : public Backend {
public:
    explicit DcuBackend(int device_index) : device_index_(device_index) {
        check_hip(hipSetDevice(device_index_), "hipSetDevice");
        check_hip(hipStreamCreate(&stream_), "hipStreamCreate");
        try {
            check_blas(hipblasCreate(&blas_), "hipblasCreate");
            check_blas(hipblasSetStream(blas_, stream_), "hipblasSetStream");
        } catch (...) {
            hipStreamDestroy(stream_);
            throw;
        }
    }

    ~DcuBackend() override {
        if (blas_) hipblasDestroy(blas_);
        if (stream_) hipStreamDestroy(stream_);
    }

    DeviceType type() const noexcept override { return DeviceType::DCU; }

    void* allocate(std::size_t bytes) override {
        activate();
        void* ptr = nullptr;
        check_hip(hipMalloc(&ptr, bytes), "hipMalloc");
        return ptr;
    }

    void deallocate(void* ptr) noexcept override {
        if (ptr) {
            hipSetDevice(device_index_);
            hipFree(ptr);
        }
    }

    void copy_from_host(void* dst, const float* src, std::size_t count) override {
        activate();
        check_hip(hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyHostToDevice, stream_),
                  "hipMemcpyAsync H2D");
        synchronize();
    }

    void copy_to_host(float* dst, const void* src, std::size_t count) override {
        activate();
        check_hip(hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDeviceToHost, stream_),
                  "hipMemcpyAsync D2H");
        synchronize();
    }

    void copy(float* dst, const float* src, std::size_t count) override {
        activate();
        check_hip(hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDeviceToDevice, stream_),
                  "hipMemcpyAsync D2D");
    }

    void synchronize() override {
        activate();
        check_hip(hipStreamSynchronize(stream_), "hipStreamSynchronize");
    }

    void fill(float* out, float value, std::size_t n) override {
        activate();
        constexpr unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
        hipLaunchKernelGGL(fill_kernel, dim3(blocks), dim3(threads), 0, stream_, out, value, n);
        check_hip(hipGetLastError(), "fill_kernel");
    }

    void add(const float* a, const float* b, float* out, std::size_t n) override {
        activate();
        constexpr unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
        hipLaunchKernelGGL(add_kernel, dim3(blocks), dim3(threads), 0, stream_, a, b, out, n);
        check_hip(hipGetLastError(), "add_kernel");
    }

    void gelu(const float* x, float* out, std::size_t n) override {
        activate();
        constexpr unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
        hipLaunchKernelGGL(gelu_kernel, dim3(blocks), dim3(threads), 0, stream_, x, out, n);
        check_hip(hipGetLastError(), "gelu_kernel");
    }

    void matmul(const float* a, const float* b, float* out,
                int m, int n, int k) override {
        activate();
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        // hipBLAS is column-major. Swapping A/B computes row-major C=A*B.
        check_blas(hipblasSgemm(blas_, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                n, m, k, &alpha, b, n, a, k, &beta, out, n),
                   "hipblasSgemm");
    }

    void matmul_backward_left(const float* grad, const float* b, float* out,
                              int m, int n, int k) override {
        activate();
        constexpr unsigned threads = 256;
        const std::size_t count = static_cast<std::size_t>(m) * k;
        const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
        hipLaunchKernelGGL(matmul_backward_left_kernel, dim3(blocks), dim3(threads), 0, stream_,
                           grad, b, out, m, n, k);
        check_hip(hipGetLastError(), "matmul_backward_left_kernel");
    }

    void matmul_backward_right(const float* a, const float* grad, float* out,
                               int m, int n, int k) override {
        activate();
        constexpr unsigned threads = 256;
        const std::size_t count = static_cast<std::size_t>(k) * n;
        const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
        hipLaunchKernelGGL(matmul_backward_right_kernel, dim3(blocks), dim3(threads), 0, stream_,
                           a, grad, out, m, n, k);
        check_hip(hipGetLastError(), "matmul_backward_right_kernel");
    }

    void softmax_rows(const float* x, float* out, int rows, int cols) override {
        activate();
        hipLaunchKernelGGL(softmax_rows_kernel, dim3(rows), dim3(1), 0, stream_, x, out, cols);
        check_hip(hipGetLastError(), "softmax_rows_kernel");
    }

    void layer_norm(const float* x, const float* weight, const float* bias,
                    float* out, int rows, int cols, float eps) override {
        activate();
        hipLaunchKernelGGL(layer_norm_kernel, dim3(rows), dim3(1), 0, stream_,
                           x, weight, bias, out, cols, eps);
        check_hip(hipGetLastError(), "layer_norm_kernel");
    }

    float squared_l2_norm(const float* x, std::size_t n) override {
        activate();
        float* device_result = nullptr;
        check_hip(hipMalloc(&device_result, sizeof(float)), "hipMalloc squared norm");
        try {
            check_hip(hipMemsetAsync(device_result, 0, sizeof(float), stream_),
                      "hipMemsetAsync squared norm");
            constexpr unsigned threads = 256;
            const unsigned needed = static_cast<unsigned>((n + threads - 1) / threads);
            const unsigned blocks = needed < 256 ? needed : 256;
            hipLaunchKernelGGL(squared_l2_norm_kernel, dim3(blocks), dim3(threads), 0, stream_,
                               x, device_result, n);
            check_hip(hipGetLastError(), "squared_l2_norm_kernel");
            float result = 0.0f;
            check_hip(hipMemcpyAsync(&result, device_result, sizeof(float),
                                     hipMemcpyDeviceToHost, stream_),
                      "hipMemcpyAsync squared norm");
            synchronize();
            hipFree(device_result);
            return result;
        } catch (...) {
            hipFree(device_result);
            throw;
        }
    }

    void adamw_update(float* values, const float* gradients,
                      float* first_moment, float* second_moment, std::size_t n,
                      float beta1, float beta2, float correction1, float correction2,
                      float learning_rate, float epsilon, float weight_decay,
                      float gradient_scale) override {
        activate();
        constexpr unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
        hipLaunchKernelGGL(adamw_kernel, dim3(blocks), dim3(threads), 0, stream_,
                           values, gradients, first_moment, second_moment, n,
                           beta1, beta2, correction1, correction2,
                           learning_rate, epsilon, weight_decay, gradient_scale);
        check_hip(hipGetLastError(), "adamw_kernel");
    }

    void cross_entropy_forward(const float* logits, const float* targets,
                               float* probabilities, float* loss, int rows, int cols) override {
        activate();
        check_hip(hipMemsetAsync(loss, 0, sizeof(float), stream_),
                  "hipMemsetAsync cross entropy loss");
        hipLaunchKernelGGL(cross_entropy_forward_kernel, dim3(rows), dim3(1), 0, stream_,
                           logits, targets, probabilities, loss, rows, cols);
        check_hip(hipGetLastError(), "cross_entropy_forward_kernel");
    }

    void cross_entropy_backward(const float* probabilities, const float* targets,
                                const float* upstream, float* grad,
                                int rows, int cols) override {
        activate();
        constexpr unsigned threads = 256;
        const std::size_t count = static_cast<std::size_t>(rows) * cols;
        const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
        hipLaunchKernelGGL(cross_entropy_backward_kernel, dim3(blocks), dim3(threads), 0, stream_,
                           probabilities, targets, upstream, grad, rows, cols);
        check_hip(hipGetLastError(), "cross_entropy_backward_kernel");
    }

private:
    void activate() const { check_hip(hipSetDevice(device_index_), "hipSetDevice"); }

    int device_index_;
    hipStream_t stream_{nullptr};
    hipblasHandle_t blas_{nullptr};
};

}  // namespace

std::shared_ptr<Backend> make_dcu_backend(int device_index) {
    return std::make_shared<DcuBackend>(device_index);
}

bool dcu_backend_available() noexcept {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
}

}  // namespace tinydl
