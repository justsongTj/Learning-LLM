#include "tinydl/optimizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tinydl {

AdamW::AdamW(std::vector<NamedParameter> parameters, AdamWOptions options)
    : options_(options) {
    if (options_.learning_rate <= 0.0f) throw std::invalid_argument("learning rate must be positive");
    states_.reserve(parameters.size());
    for (auto& entry : parameters) {
        Tensor first(entry.second->value().shape(), entry.second->value().backend());
        Tensor second(entry.second->value().shape(), entry.second->value().backend());
        first.fill(0.0f);
        second.fill(0.0f);
        states_.push_back({entry.second, std::move(first), std::move(second)});
    }
}

void AdamW::step() {
    ++step_;
    double squared_norm = 0.0;
    for (const auto& state : states_) {
        if (const Tensor* grad = state.parameter->grad())
            squared_norm += state.parameter->value().backend()->squared_l2_norm(
                grad->data(), grad->numel());
    }
    const float norm = static_cast<float>(std::sqrt(squared_norm));
    const float clip = options_.max_grad_norm > 0.0f && norm > options_.max_grad_norm
        ? options_.max_grad_norm / (norm + 1.0e-6f) : 1.0f;
    const float correction1 = 1.0f - std::pow(options_.beta1, static_cast<float>(step_));
    const float correction2 = 1.0f - std::pow(options_.beta2, static_cast<float>(step_));

    for (auto& state : states_) {
        const Tensor* grad_tensor = state.parameter->grad();
        if (!grad_tensor) continue;
        Tensor& values = state.parameter->value();
        values.backend()->adamw_update(
            values.data(), grad_tensor->data(), state.first_moment.data(), state.second_moment.data(),
            values.numel(), options_.beta1, options_.beta2, correction1, correction2,
            options_.learning_rate, options_.epsilon, options_.weight_decay, clip);
    }
}

void AdamW::zero_grad() {
    for (auto& state : states_) state.parameter->zero_grad();
}

void AdamW::set_learning_rate(float learning_rate) {
    if (learning_rate <= 0.0f) throw std::invalid_argument("learning rate must be positive");
    options_.learning_rate = learning_rate;
}

}  // namespace tinydl
