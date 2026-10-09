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
        const std::size_t count = entry.second->value().numel();
        states_.push_back({entry.second, std::vector<float>(count, 0.0f),
                           std::vector<float>(count, 0.0f)});
    }
}

void AdamW::step() {
    ++step_;
    double squared_norm = 0.0;
    for (const auto& state : states_) {
        if (const Tensor* grad = state.parameter->grad())
            for (float value : grad->to_vector()) squared_norm += static_cast<double>(value) * value;
    }
    const float norm = static_cast<float>(std::sqrt(squared_norm));
    const float clip = options_.max_grad_norm > 0.0f && norm > options_.max_grad_norm
        ? options_.max_grad_norm / (norm + 1.0e-6f) : 1.0f;
    const float correction1 = 1.0f - std::pow(options_.beta1, static_cast<float>(step_));
    const float correction2 = 1.0f - std::pow(options_.beta2, static_cast<float>(step_));

    for (auto& state : states_) {
        const Tensor* grad_tensor = state.parameter->grad();
        if (!grad_tensor) continue;
        auto values = state.parameter->value().to_vector();
        const auto gradients = grad_tensor->to_vector();
        for (std::size_t i = 0; i < values.size(); ++i) {
            const float gradient = gradients[i] * clip;
            state.first_moment[i] = options_.beta1 * state.first_moment[i] +
                                    (1.0f - options_.beta1) * gradient;
            state.second_moment[i] = options_.beta2 * state.second_moment[i] +
                                     (1.0f - options_.beta2) * gradient * gradient;
            const float first = state.first_moment[i] / correction1;
            const float second = state.second_moment[i] / correction2;
            values[i] -= options_.learning_rate *
                (first / (std::sqrt(second) + options_.epsilon) + options_.weight_decay * values[i]);
        }
        state.parameter->value().copy_from(values);
    }
}

void AdamW::zero_grad() {
    for (auto& state : states_) state.parameter->zero_grad();
}

}  // namespace tinydl
