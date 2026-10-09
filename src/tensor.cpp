#include "tinydl/tensor.h"

#include <numeric>
#include <stdexcept>

namespace tinydl {
namespace {

void require_same_backend(const Tensor& a, const Tensor& b) {
    if (a.backend().get() != b.backend().get()) {
        throw std::invalid_argument("operands must use the same backend instance");
    }
}

}  // namespace

Tensor::Tensor(std::vector<std::size_t> shape, std::shared_ptr<Backend> backend)
    : shape_(std::move(shape)), backend_(std::move(backend)) {
    if (!backend_) throw std::invalid_argument("backend must not be null");
    if (shape_.empty()) throw std::invalid_argument("scalar tensors are not supported yet");
    numel_ = std::accumulate(shape_.begin(), shape_.end(), std::size_t{1},
                             std::multiplies<std::size_t>{});
    if (numel_ == 0) throw std::invalid_argument("zero-sized tensors are not supported yet");
    void* ptr = backend_->allocate(numel_ * sizeof(float));
    auto owner = backend_;
    storage_ = std::shared_ptr<void>(ptr, [owner](void* p) { owner->deallocate(p); });
}

Tensor::Tensor(std::initializer_list<std::size_t> shape, std::shared_ptr<Backend> backend)
    : Tensor(std::vector<std::size_t>(shape), std::move(backend)) {}

void Tensor::copy_from(const std::vector<float>& values) {
    if (values.size() != numel_) throw std::invalid_argument("copy size does not match tensor");
    backend_->copy_from_host(data(), values.data(), numel_);
}

std::vector<float> Tensor::to_vector() const {
    std::vector<float> result(numel_);
    backend_->copy_to_host(result.data(), data(), numel_);
    return result;
}

void Tensor::fill(float value) { backend_->fill(data(), value, numel_); }

Tensor to_device(const Tensor& source, std::shared_ptr<Backend> target) {
    Tensor result(source.shape(), std::move(target));
    result.copy_from(source.to_vector());
    return result;
}

Tensor add(const Tensor& a, const Tensor& b) {
    require_same_backend(a, b);
    if (a.shape() != b.shape()) throw std::invalid_argument("add requires equal shapes");
    Tensor out(a.shape(), a.backend());
    a.backend()->add(a.data(), b.data(), out.data(), a.numel());
    return out;
}

Tensor gelu(const Tensor& x) {
    Tensor out(x.shape(), x.backend());
    x.backend()->gelu(x.data(), out.data(), x.numel());
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    require_same_backend(a, b);
    if (a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul expects [M,K] x [K,N]");
    }
    const int m = static_cast<int>(a.shape()[0]);
    const int k = static_cast<int>(a.shape()[1]);
    const int n = static_cast<int>(b.shape()[1]);
    Tensor out({static_cast<std::size_t>(m), static_cast<std::size_t>(n)}, a.backend());
    a.backend()->matmul(a.data(), b.data(), out.data(), m, n, k);
    return out;
}

Tensor softmax_rows(const Tensor& x) {
    if (x.ndim() != 2) throw std::invalid_argument("softmax_rows expects a matrix");
    Tensor out(x.shape(), x.backend());
    x.backend()->softmax_rows(x.data(), out.data(), static_cast<int>(x.shape()[0]),
                              static_cast<int>(x.shape()[1]));
    return out;
}

Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
    require_same_backend(x, weight);
    require_same_backend(x, bias);
    if (x.ndim() != 2 || weight.ndim() != 1 || bias.ndim() != 1 ||
        weight.numel() != x.shape()[1] || bias.numel() != x.shape()[1]) {
        throw std::invalid_argument("layer_norm expects x=[rows,cols], weight=bias=[cols]");
    }
    Tensor out(x.shape(), x.backend());
    x.backend()->layer_norm(x.data(), weight.data(), bias.data(), out.data(),
                            static_cast<int>(x.shape()[0]), static_cast<int>(x.shape()[1]), eps);
    return out;
}

}  // namespace tinydl
