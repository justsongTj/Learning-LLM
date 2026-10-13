#pragma once

#include "tinydl/device.h"

#include <cstddef>
#include <memory>

namespace tinydl {

class Backend {
public:
    virtual ~Backend() = default;

    virtual DeviceType type() const noexcept = 0;
    virtual void* allocate(std::size_t bytes) = 0;
    virtual void deallocate(void* ptr) noexcept = 0;
    virtual void copy_from_host(void* dst, const float* src, std::size_t count) = 0;
    virtual void copy_to_host(float* dst, const void* src, std::size_t count) = 0;
    virtual void copy(float* dst, const float* src, std::size_t count) = 0;
    virtual void synchronize() = 0;

    virtual void fill(float* out, float value, std::size_t n) = 0;
    virtual void add(const float* a, const float* b, float* out, std::size_t n) = 0;
    virtual void gelu(const float* x, float* out, std::size_t n) = 0;
    virtual void matmul(const float* a, const float* b, float* out,
                        int m, int n, int k) = 0;
    virtual void matmul_backward_left(const float* grad, const float* b, float* out,
                                      int m, int n, int k) = 0;
    virtual void matmul_backward_right(const float* a, const float* grad, float* out,
                                       int m, int n, int k) = 0;
    virtual void softmax_rows(const float* x, float* out, int rows, int cols) = 0;
    virtual void layer_norm(const float* x, const float* weight, const float* bias,
                            float* out, int rows, int cols, float eps) = 0;
    virtual float squared_l2_norm(const float* x, std::size_t n) = 0;
    virtual void adamw_update(float* values, const float* gradients,
                              float* first_moment, float* second_moment, std::size_t n,
                              float beta1, float beta2, float correction1, float correction2,
                              float learning_rate, float epsilon, float weight_decay,
                              float gradient_scale) = 0;
    virtual void cross_entropy_forward(const float* logits, const float* targets,
                                       float* probabilities, float* loss, int rows, int cols) = 0;
    virtual void cross_entropy_backward(const float* probabilities, const float* targets,
                                        const float* upstream, float* grad,
                                        int rows, int cols) = 0;
};

std::shared_ptr<Backend> make_cpu_backend();
std::shared_ptr<Backend> make_dcu_backend(int device_index = 0);
bool dcu_backend_available() noexcept;

}  // namespace tinydl
