#include "tinydl/tensor.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

void print_matrix(const tinydl::Tensor& tensor, const std::string& name) {
    const auto values = tensor.to_vector();
    const auto& shape = tensor.shape();
    std::cout << name << " [" << shape[0] << ", " << shape[1] << "]\n";
    for (std::size_t row = 0; row < shape[0]; ++row) {
        for (std::size_t col = 0; col < shape[1]; ++col) {
            std::cout << std::setw(11) << std::setprecision(6)
                      << values[row * shape[1] + col] << ' ';
        }
        std::cout << '\n';
    }
}

tinydl::Tensor run_block(const std::shared_ptr<tinydl::Backend>& backend) {
    tinydl::Tensor x({2, 3}, backend);
    tinydl::Tensor weight({3, 4}, backend);
    tinydl::Tensor residual({2, 4}, backend);
    tinydl::Tensor gamma({4}, backend);
    tinydl::Tensor beta({4}, backend);

    x.copy_from({1.0f, -2.0f, 0.5f, 0.25f, 1.5f, -1.0f});
    weight.copy_from({0.2f, 0.4f, -0.5f, 0.1f,
                      0.7f, -0.3f, 0.8f, 0.2f,
                      -0.6f, 0.5f, 0.1f, -0.4f});
    residual.copy_from({0.1f, 0.1f, 0.1f, 0.1f,
                        -0.1f, -0.1f, -0.1f, -0.1f});
    gamma.copy_from({1.0f, 1.0f, 1.0f, 1.0f});
    beta.copy_from({0.0f, 0.0f, 0.0f, 0.0f});

    auto hidden = tinydl::gelu(tinydl::matmul(x, weight));
    auto normalized = tinydl::layer_norm(tinydl::add(hidden, residual), gamma, beta);
    return tinydl::softmax_rows(normalized);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool use_dcu = argc > 1 && std::string(argv[1]) == "--dcu";
        auto backend = use_dcu ? tinydl::make_dcu_backend() : tinydl::make_cpu_backend();
        print_matrix(run_block(backend), use_dcu ? "DCU output" : "CPU output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TinyDL demo failed: " << error.what() << '\n';
        return 1;
    }
}
