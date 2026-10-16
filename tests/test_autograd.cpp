#include "tinydl/autograd.h"
#include "tinydl/optimizer.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_close(const std::vector<float>& actual, const std::vector<float>& expected,
                  float tolerance, const std::string& name) {
    if (actual.size() != expected.size()) throw std::runtime_error(name + ": size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::fabs(actual[i] - expected[i]) > tolerance)
            throw std::runtime_error(name + ": mismatch at " + std::to_string(i));
}

}  // namespace

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
        expect_close(input.grad()->to_vector(),
                     {-0.00501928f, -0.00585615f, 0.749992f, 0.874991f},
                     1.0e-5f, "input gradient");
        expect_close(weight.grad()->to_vector(),
                     {1.496638f, -1.496638f, 1.993286f, -1.993286f},
                     1.0e-5f, "weight gradient");

        tinydl::Tensor masked_logits_tensor({2, 2}, cpu);
        masked_logits_tensor.copy_from({2.0f, 0.0f, 0.0f, 2.0f});
        tinydl::Variable masked_logits(std::move(masked_logits_tensor), true);
        auto masked_loss = tinydl::cross_entropy(masked_logits, {0, 0}, {1.0f, 0.0f});
        expect_close(masked_loss.value().to_vector(), {0.126928f},
                     1.0e-5f, "masked cross entropy loss");
        masked_loss.backward();
        if (!masked_logits.grad()) throw std::runtime_error("missing masked loss gradient");
        expect_close(masked_logits.grad()->to_vector(),
                     {-0.119203f, 0.119203f, 0.0f, 0.0f},
                     1.0e-5f, "masked cross entropy gradient");

        bool rejected_empty_mask = false;
        try {
            static_cast<void>(tinydl::cross_entropy(masked_logits, {0, 0}, {0.0f, 0.0f}));
        } catch (const std::invalid_argument&) {
            rejected_empty_mask = true;
        }
        if (!rejected_empty_mask)
            throw std::runtime_error("cross entropy accepted an empty loss mask");

        tinydl::Tensor parameter_tensor({2}, cpu);
        tinydl::Tensor gradient_tensor({2}, cpu);
        parameter_tensor.copy_from({1.0f, 2.0f});
        gradient_tensor.copy_from({0.1f, -0.2f});
        tinydl::Variable parameter(std::move(parameter_tensor), true);
        parameter.accumulate_grad(gradient_tensor);
        tinydl::AdamWOptions options;
        options.learning_rate = 0.01f;
        options.weight_decay = 0.01f;
        options.max_grad_norm = 0.0f;
        tinydl::AdamW optimizer({{"parameter", &parameter}}, options);
        optimizer.step();
        expect_close(parameter.value().to_vector(), {0.9899f, 2.0098f},
                     1.0e-5f, "AdamW update");

        tinydl::Tensor accumulated_parameter_tensor({2}, cpu);
        accumulated_parameter_tensor.copy_from({1.0f, 2.0f});
        tinydl::Variable accumulated_parameter(std::move(accumulated_parameter_tensor), true);
        accumulated_parameter.accumulate_grad(gradient_tensor);
        accumulated_parameter.accumulate_grad(gradient_tensor);
        tinydl::AdamW accumulated_optimizer(
            {{"parameter", &accumulated_parameter}}, options);
        accumulated_optimizer.step(0.5f);
        expect_close(accumulated_parameter.value().to_vector(), {0.9899f, 2.0098f},
                     1.0e-5f, "scaled accumulated AdamW update");
        std::cout << "TinyDL autograd smoke test passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
