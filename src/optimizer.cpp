#include "tinydl/optimizer.h"

#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace tinydl {
namespace {

template <typename T>
void write_value(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_value(std::istream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) throw std::runtime_error("truncated AdamW state");
    return value;
}

}  // namespace

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

void AdamW::save_state(std::ostream& stream) const {
    write_value(stream, step_);
    write_value(stream, static_cast<std::uint64_t>(states_.size()));
    for (const auto& state : states_) {
        const auto first = state.first_moment.to_vector();
        const auto second = state.second_moment.to_vector();
        write_value(stream, static_cast<std::uint64_t>(first.size()));
        stream.write(reinterpret_cast<const char*>(first.data()),
                     static_cast<std::streamsize>(first.size() * sizeof(float)));
        stream.write(reinterpret_cast<const char*>(second.data()),
                     static_cast<std::streamsize>(second.size() * sizeof(float)));
    }
    if (!stream) throw std::runtime_error("failed while writing AdamW state");
}

void AdamW::load_state(std::istream& stream) {
    step_ = read_value<std::uint64_t>(stream);
    const auto count = read_value<std::uint64_t>(stream);
    if (count != states_.size()) throw std::runtime_error("AdamW parameter count mismatch");
    for (auto& state : states_) {
        const auto elements = read_value<std::uint64_t>(stream);
        if (elements != state.first_moment.numel())
            throw std::runtime_error("AdamW parameter shape mismatch");
        std::vector<float> first(static_cast<std::size_t>(elements));
        std::vector<float> second(static_cast<std::size_t>(elements));
        stream.read(reinterpret_cast<char*>(first.data()),
                    static_cast<std::streamsize>(first.size() * sizeof(float)));
        stream.read(reinterpret_cast<char*>(second.data()),
                    static_cast<std::streamsize>(second.size() * sizeof(float)));
        if (!stream) throw std::runtime_error("truncated AdamW moment tensors");
        state.first_moment.copy_from(first);
        state.second_moment.copy_from(second);
    }
}

}  // namespace tinydl
