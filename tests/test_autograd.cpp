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

        tinydl::Tensor attention_q({3, 4}, cpu);
        tinydl::Tensor attention_k({3, 4}, cpu);
        tinydl::Tensor attention_v({3, 4}, cpu);
        tinydl::Tensor attention_probabilities({2, 3, 3}, cpu);
        tinydl::Tensor attention_output({3, 4}, cpu);
        tinydl::Tensor attention_upstream({3, 4}, cpu);
        tinydl::Tensor attention_dq({3, 4}, cpu);
        tinydl::Tensor attention_dk({3, 4}, cpu);
        tinydl::Tensor attention_dv({3, 4}, cpu);
        attention_q.copy_from({0.2f, -0.1f, 0.4f, 0.3f,
                               0.5f, 0.2f, -0.3f, 0.1f,
                               -0.4f, 0.6f, 0.2f, -0.5f});
        attention_k.copy_from({0.1f, 0.7f, -0.2f, 0.5f,
                               0.3f, -0.4f, 0.6f, 0.2f,
                               0.8f, 0.1f, -0.1f, 0.9f});
        attention_v.copy_from({1.0f, 2.0f, 3.0f, 4.0f,
                               -1.0f, 0.5f, 2.0f, -2.0f,
                               0.25f, -0.75f, 1.5f, 0.8f});
        attention_upstream.copy_from({0.3f, -0.2f, 0.1f, 0.4f,
                                      -0.5f, 0.7f, -0.3f, 0.2f,
                                      0.6f, 0.1f, -0.4f, 0.8f});
        cpu->causal_attention_forward(
            attention_q.data(), attention_k.data(), attention_v.data(),
            attention_probabilities.data(), attention_output.data(), 3, 4, 2);
        expect_close(attention_probabilities.to_vector(),
                     {1.0f, 0.0f, 0.0f, 0.52120048f, 0.47879952f, 0.0f,
                      0.44871391f, 0.26590165f, 0.28538444f,
                      1.0f, 0.0f, 0.0f, 0.54758526f, 0.45241474f, 0.0f,
                      0.31994145f, 0.39835402f, 0.28170454f},
                     1.0e-5f, "causal attention probabilities");
        expect_close(attention_output.to_vector(),
                     {1.0f, 2.0f, 3.0f, 4.0f,
                      0.04240097f, 1.28180073f, 2.54758526f, 1.28551153f,
                      0.25415837f, 0.81634032f, 2.17908918f, 0.70842138f},
                     1.0e-5f, "causal attention output");
        cpu->causal_attention_backward(
            attention_q.data(), attention_k.data(), attention_v.data(),
            attention_probabilities.data(), attention_upstream.data(),
            attention_dq.data(), attention_dk.data(), attention_dv.data(), 3, 4, 2);
        expect_close(attention_dq.to_vector(),
                     {0.0f, 0.0f, 0.0f, 0.0f,
                      -0.00176459f, 0.00970524f, -0.12612640f, 0.04729740f,
                      -0.05196479f, 0.18144297f, -0.46524636f, 0.20452457f},
                     1.0e-5f, "causal attention q gradient");
        expect_close(attention_dk.to_vector(),
                     {-0.06740630f, 0.10949125f, 0.05699132f, -0.24495599f,
                      0.05456158f, -0.09022416f, -0.07073174f, 0.27930706f,
                      0.01284472f, -0.01926709f, 0.01374043f, -0.03435107f},
                     1.0e-5f, "causal attention k gradient");
        expect_close(attention_dv.to_vector(),
                     {0.30862811f, 0.20971173f, -0.19225216f, 0.76547021f,
                      -0.07985877f, 0.36174983f, -0.29506603f, 0.40916616f,
                      0.17123066f, 0.02853844f, -0.11268181f, 0.22536363f},
                     1.0e-5f, "causal attention v gradient");

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
