#pragma once

#include "tinydl/tensor.h"

#include <functional>
#include <memory>
#include <vector>

namespace tinydl {

class Variable {
public:
    struct Node;

    Variable() = default;
    explicit Variable(Tensor value, bool requires_grad = false);

    const Tensor& value() const;
    Tensor& value();
    const Tensor* grad() const;
    bool requires_grad() const;

    void backward();
    void zero_grad();
    void accumulate_grad(const Tensor& gradient) const;

    static Variable from_graph(Tensor value, bool requires_grad,
                               std::vector<Variable> parents,
                               std::function<void(const Tensor&)> backward);

private:
    explicit Variable(std::shared_ptr<Node> node);
    std::shared_ptr<Node> node_;

    friend Variable add(const Variable&, const Variable&);
    friend Variable add_bias(const Variable&, const Variable&);
    friend Variable matmul(const Variable&, const Variable&);
    friend Variable gelu(const Variable&);
    friend Variable layer_norm(const Variable&, const Variable&, const Variable&, float);
    friend Variable embedding(const Variable&, const std::vector<int>&);
    friend Variable causal_self_attention(const Variable&, const Variable&, const Variable&,
                                           int);
    friend Variable cross_entropy(const Variable&, const std::vector<int>&);
};

Variable add(const Variable& a, const Variable& b);
Variable add_bias(const Variable& matrix, const Variable& bias);
Variable matmul(const Variable& a, const Variable& b);
Variable gelu(const Variable& x);
Variable layer_norm(const Variable& x, const Variable& weight, const Variable& bias,
                    float eps = 1.0e-5f);
Variable embedding(const Variable& table, const std::vector<int>& token_ids);
Variable causal_self_attention(const Variable& q, const Variable& k, const Variable& v,
                               int num_heads);
Variable cross_entropy(const Variable& logits, const std::vector<int>& targets);

}  // namespace tinydl
