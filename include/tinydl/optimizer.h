#pragma once

#include "tinydl/nn.h"

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace tinydl {

struct AdamWOptions {
    float learning_rate{3.0e-4f};
    float beta1{0.9f};
    float beta2{0.999f};
    float epsilon{1.0e-8f};
    float weight_decay{0.01f};
    float max_grad_norm{1.0f};
};

class AdamW {
public:
    AdamW(std::vector<NamedParameter> parameters, AdamWOptions options = {});
    void step(float gradient_scale = 1.0f);
    void zero_grad();
    void set_learning_rate(float learning_rate);
    float learning_rate() const noexcept { return options_.learning_rate; }
    std::uint64_t step_count() const noexcept { return step_; }
    void save_state(std::ostream& stream) const;
    void load_state(std::istream& stream);

private:
    struct State {
        Variable* parameter{};
        Tensor first_moment;
        Tensor second_moment;
    };

    std::vector<State> states_;
    AdamWOptions options_;
    std::uint64_t step_{0};
};

}  // namespace tinydl
