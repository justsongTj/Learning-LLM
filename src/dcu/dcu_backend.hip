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
