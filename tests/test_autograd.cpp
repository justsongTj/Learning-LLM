#include "tinydl/autograd.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        auto cpu = tinydl::make_cpu_backend();
        tinydl::Tensor input_tensor({2, 2}, cpu);
        tinydl::Tensor weight_tensor({2, 2}, cpu);
        input_tensor.copy_from({1.0f, 2.0f, 3.0f, 4.0f});
        weight_tensor.copy_from({0.5f, -1.0f, 2.0f, 0.25f});
        tinydl::Variable input(std::move(input_tensor), true);
        tinydl::Variable weight(std::move(weight_tensor), true);
        auto logits = tinydl::matmul(input, weight);
        auto loss = tinydl::cross_entropy(logits, {0, 1});
        loss.backward();
        if (!input.grad() || !weight.grad()) throw std::runtime_error("missing matmul gradient");
        for (float value : input.grad()->to_vector())
            if (!std::isfinite(value)) throw std::runtime_error("non-finite input gradient");
        for (float value : weight.grad()->to_vector())
            if (!std::isfinite(value)) throw std::runtime_error("non-finite weight gradient");
        std::cout << "TinyDL autograd smoke test passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
