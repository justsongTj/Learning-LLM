#pragma once

#include "tinydl/backend.h"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

namespace tinydl {

class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<std::size_t> shape, std::shared_ptr<Backend> backend);
    Tensor(std::initializer_list<std::size_t> shape, std::shared_ptr<Backend> backend);

    float* data() noexcept { return static_cast<float*>(storage_.get()); }
    const float* data() const noexcept { return static_cast<const float*>(storage_.get()); }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    std::size_t numel() const noexcept { return numel_; }
    std::size_t ndim() const noexcept { return shape_.size(); }
    const std::shared_ptr<Backend>& backend() const noexcept { return backend_; }
    DeviceType device_type() const noexcept { return backend_->type(); }

    void copy_from(const std::vector<float>& values);
    std::vector<float> to_vector() const;
    void fill(float value);

private:
    std::vector<std::size_t> shape_;
    std::size_t numel_{0};
    std::shared_ptr<Backend> backend_;
    std::shared_ptr<void> storage_;
};

Tensor to_device(const Tensor& source, std::shared_ptr<Backend> target);
Tensor add(const Tensor& a, const Tensor& b);
Tensor gelu(const Tensor& x);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor softmax_rows(const Tensor& x);
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  float eps = 1.0e-5f);

}  // namespace tinydl
