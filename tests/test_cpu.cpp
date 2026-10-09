#include "tinydl/tensor.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_close(const std::vector<float>& actual, const std::vector<float>& expected,
                  float tolerance, const std::string& test_name) {
    if (actual.size() != expected.size()) throw std::runtime_error(test_name + ": size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error(test_name + ": mismatch at element " + std::to_string(i));
        }
    }
}

void test_matmul(const std::shared_ptr<tinydl::Backend>& cpu) {
    tinydl::Tensor a({2, 3}, cpu);
    tinydl::Tensor b({3, 2}, cpu);
    a.copy_from({1, 2, 3, 4, 5, 6});
    b.copy_from({7, 8, 9, 10, 11, 12});
    expect_close(tinydl::matmul(a, b).to_vector(), {58, 64, 139, 154}, 1.0e-6f, "matmul");
}

void test_softmax(const std::shared_ptr<tinydl::Backend>& cpu) {
    tinydl::Tensor x({2, 3}, cpu);
    x.copy_from({1, 1, 1, 1001, 1001, 1001});
    const auto result = tinydl::softmax_rows(x).to_vector();
    expect_close(result, {1.0f / 3, 1.0f / 3, 1.0f / 3,
                          1.0f / 3, 1.0f / 3, 1.0f / 3}, 1.0e-6f, "softmax");
}

void test_layer_norm(const std::shared_ptr<tinydl::Backend>& cpu) {
    tinydl::Tensor x({1, 3}, cpu);
    tinydl::Tensor weight({3}, cpu);
    tinydl::Tensor bias({3}, cpu);
    x.copy_from({1, 2, 3});
    weight.copy_from({1, 1, 1});
    bias.copy_from({0, 0, 0});
    expect_close(tinydl::layer_norm(x, weight, bias).to_vector(),
                 {-1.2247356f, 0.0f, 1.2247356f}, 1.0e-5f, "layer_norm");
}

}  // namespace

int main() {
    try {
        auto cpu = tinydl::make_cpu_backend();
        test_matmul(cpu);
        test_softmax(cpu);
        test_layer_norm(cpu);
        std::cout << "All TinyDL CPU tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
