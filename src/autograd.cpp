#include "tinydl/autograd.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace tinydl {

struct Variable::Node {
    explicit Node(Tensor tensor, bool needs_grad)
        : value(std::move(tensor)), requires_grad(needs_grad) {}

    Tensor value;
    std::unique_ptr<Tensor> gradient;
    bool requires_grad{false};
    std::vector<Variable> parents;
    std::function<void(const Tensor&)> backward_fn;
};

namespace {

Tensor tensor_from(const std::vector<std::size_t>& shape,
                   const std::shared_ptr<Backend>& backend,
                   const std::vector<float>& values) {
    Tensor result(shape, backend);
    result.copy_from(values);
    return result;
}

}  // namespace

Variable::Variable(Tensor value, bool requires_grad)
    : node_(std::make_shared<Node>(std::move(value), requires_grad)) {}

Variable::Variable(std::shared_ptr<Node> node) : node_(std::move(node)) {}

const Tensor& Variable::value() const {
    if (!node_) throw std::logic_error("empty Variable");
    return node_->value;
}

Tensor& Variable::value() {
    if (!node_) throw std::logic_error("empty Variable");
    return node_->value;
}

const Tensor* Variable::grad() const { return node_ && node_->gradient ? node_->gradient.get() : nullptr; }
bool Variable::requires_grad() const { return node_ && node_->requires_grad; }

Variable Variable::from_graph(Tensor value, bool requires_grad,
                              std::vector<Variable> parents,
                              std::function<void(const Tensor&)> backward) {
    auto node = std::make_shared<Node>(std::move(value), requires_grad);
    node->parents = std::move(parents);
    node->backward_fn = std::move(backward);
    return Variable(std::move(node));
}

void Variable::accumulate_grad(const Tensor& gradient) const {
    if (!requires_grad()) return;
    if (gradient.shape() != value().shape()) throw std::invalid_argument("gradient shape mismatch");
    if (!node_->gradient) {
        node_->gradient = std::make_unique<Tensor>(gradient.shape(), value().backend());
        value().backend()->copy(node_->gradient->data(), gradient.data(), gradient.numel());
        return;
    }
    value().backend()->add(node_->gradient->data(), gradient.data(),
                           node_->gradient->data(), gradient.numel());
}

void Variable::zero_grad() {
    if (node_) node_->gradient.reset();
}

void Variable::backward() {
    if (value().numel() != 1) throw std::logic_error("backward requires a scalar output");
    Tensor seed(value().shape(), value().backend());
    seed.fill(1.0f);
    accumulate_grad(seed);

    std::unordered_set<const Node*> seen;
    std::vector<Variable> order;
    std::function<void(const Variable&)> visit = [&](const Variable& variable) {
        if (!variable.node_ || !seen.insert(variable.node_.get()).second) return;
        for (const auto& parent : variable.node_->parents) visit(parent);
        order.push_back(variable);
    };
    visit(*this);
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if (it->node_->backward_fn && it->node_->gradient) {
            it->node_->backward_fn(*it->node_->gradient);
        }
    }
}

Variable add(const Variable& a, const Variable& b) {
    Tensor output = tinydl::add(a.value(), b.value());
    const bool track = a.requires_grad() || b.requires_grad();
    return Variable::from_graph(std::move(output), track, {a, b}, [a, b](const Tensor& grad) {
        a.accumulate_grad(grad);
        b.accumulate_grad(grad);
    });
}

Variable add_bias(const Variable& matrix, const Variable& bias) {
    if (matrix.value().ndim() != 2 || bias.value().ndim() != 1 ||
        matrix.value().shape()[1] != bias.value().numel()) {
        throw std::invalid_argument("add_bias expects [rows,cols] and [cols]");
    }
    auto values = matrix.value().to_vector();
    const auto bias_values = bias.value().to_vector();
    const std::size_t cols = bias_values.size();
    for (std::size_t i = 0; i < values.size(); ++i) values[i] += bias_values[i % cols];
    Tensor output = tensor_from(matrix.value().shape(), matrix.value().backend(), values);
    const bool track = matrix.requires_grad() || bias.requires_grad();
    return Variable::from_graph(std::move(output), track, {matrix, bias},
        [matrix, bias, cols](const Tensor& grad) {
            matrix.accumulate_grad(grad);
            if (bias.requires_grad()) {
                const auto incoming = grad.to_vector();
                std::vector<float> reduced(cols, 0.0f);
                for (std::size_t i = 0; i < incoming.size(); ++i) reduced[i % cols] += incoming[i];
                bias.accumulate_grad(tensor_from({cols}, bias.value().backend(), reduced));
            }
        });
}

Variable matmul(const Variable& a, const Variable& b) {
    Tensor output = tinydl::matmul(a.value(), b.value());
    const bool track = a.requires_grad() || b.requires_grad();
    return Variable::from_graph(std::move(output), track, {a, b}, [a, b](const Tensor& grad) {
        const std::size_t m = a.value().shape()[0];
        const std::size_t k = a.value().shape()[1];
        const std::size_t n = b.value().shape()[1];
        if (a.requires_grad()) {
            Tensor da(a.value().shape(), a.value().backend());
            a.value().backend()->matmul_backward_left(
                grad.data(), b.value().data(), da.data(),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
            a.accumulate_grad(da);
        }
        if (b.requires_grad()) {
            Tensor db(b.value().shape(), b.value().backend());
            b.value().backend()->matmul_backward_right(
                a.value().data(), grad.data(), db.data(),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
            b.accumulate_grad(db);
        }
    });
}

Variable gelu(const Variable& x) {
    Tensor output = tinydl::gelu(x.value());
    return Variable::from_graph(std::move(output), x.requires_grad(), {x}, [x](const Tensor& grad) {
        if (!x.requires_grad()) return;
        const auto xv = x.value().to_vector();
        const auto gv = grad.to_vector();
        std::vector<float> dx(xv.size());
        constexpr float k = 0.7978845608028654f;
        for (std::size_t i = 0; i < xv.size(); ++i) {
            const float v = xv[i];
            const float u = k * (v + 0.044715f * v * v * v);
            const float t = std::tanh(u);
            const float derivative = 0.5f * (1.0f + t) +
                0.5f * v * (1.0f - t * t) * k * (1.0f + 3.0f * 0.044715f * v * v);
            dx[i] = gv[i] * derivative;
        }
        x.accumulate_grad(tensor_from(x.value().shape(), x.value().backend(), dx));
    });
}

Variable layer_norm(const Variable& x, const Variable& weight, const Variable& bias, float eps) {
    Tensor output = tinydl::layer_norm(x.value(), weight.value(), bias.value(), eps);
    const bool track = x.requires_grad() || weight.requires_grad() || bias.requires_grad();
    return Variable::from_graph(std::move(output), track, {x, weight, bias},
        [x, weight, bias, eps](const Tensor& grad) {
            const auto xv = x.value().to_vector();
            const auto wv = weight.value().to_vector();
            const auto gv = grad.to_vector();
            const std::size_t rows = x.value().shape()[0];
            const std::size_t cols = x.value().shape()[1];
            std::vector<float> dx(xv.size(), 0.0f), dw(cols, 0.0f), db(cols, 0.0f);
            for (std::size_t row = 0; row < rows; ++row) {
                const std::size_t offset = row * cols;
                float mean = 0.0f;
                for (std::size_t col = 0; col < cols; ++col) mean += xv[offset + col];
                mean /= static_cast<float>(cols);
                float variance = 0.0f;
                for (std::size_t col = 0; col < cols; ++col) {
                    const float centered = xv[offset + col] - mean;
                    variance += centered * centered;
                }
                variance /= static_cast<float>(cols);
                const float inv_std = 1.0f / std::sqrt(variance + eps);
                float sum_dy = 0.0f;
                float sum_dy_xhat = 0.0f;
                for (std::size_t col = 0; col < cols; ++col) {
                    const float xhat = (xv[offset + col] - mean) * inv_std;
                    const float dy = gv[offset + col] * wv[col];
                    sum_dy += dy;
                    sum_dy_xhat += dy * xhat;
                    dw[col] += gv[offset + col] * xhat;
                    db[col] += gv[offset + col];
                }
                for (std::size_t col = 0; col < cols; ++col) {
                    const float xhat = (xv[offset + col] - mean) * inv_std;
                    const float dy = gv[offset + col] * wv[col];
                    dx[offset + col] = inv_std / static_cast<float>(cols) *
                        (static_cast<float>(cols) * dy - sum_dy - xhat * sum_dy_xhat);
                }
            }
            if (x.requires_grad()) x.accumulate_grad(tensor_from(x.value().shape(), x.value().backend(), dx));
            if (weight.requires_grad()) weight.accumulate_grad(tensor_from({cols}, weight.value().backend(), dw));
            if (bias.requires_grad()) bias.accumulate_grad(tensor_from({cols}, bias.value().backend(), db));
        });
}

Variable embedding(const Variable& table, const std::vector<int>& token_ids) {
    if (table.value().ndim() != 2) throw std::invalid_argument("embedding table must be [vocab,channels]");
    const std::size_t vocab = table.value().shape()[0];
    const std::size_t channels = table.value().shape()[1];
    const auto weights = table.value().to_vector();
    std::vector<float> values(token_ids.size() * channels);
    for (std::size_t row = 0; row < token_ids.size(); ++row) {
        if (token_ids[row] < 0 || static_cast<std::size_t>(token_ids[row]) >= vocab)
            throw std::out_of_range("embedding token id is outside vocabulary");
        std::copy_n(weights.data() + static_cast<std::size_t>(token_ids[row]) * channels,
                    channels, values.data() + row * channels);
    }
    Tensor output = tensor_from({token_ids.size(), channels}, table.value().backend(), values);
    return Variable::from_graph(std::move(output), table.requires_grad(), {table},
        [table, token_ids, vocab, channels](const Tensor& grad) {
            if (!table.requires_grad()) return;
            std::vector<float> dw(vocab * channels, 0.0f);
            const auto incoming = grad.to_vector();
            for (std::size_t row = 0; row < token_ids.size(); ++row)
                for (std::size_t col = 0; col < channels; ++col)
                    dw[static_cast<std::size_t>(token_ids[row]) * channels + col] +=
                        incoming[row * channels + col];
            table.accumulate_grad(tensor_from(table.value().shape(), table.value().backend(), dw));
        });
}

Variable causal_self_attention(const Variable& q, const Variable& k, const Variable& v,
                               int num_heads) {
    if (q.value().shape() != k.value().shape() || q.value().shape() != v.value().shape() ||
        q.value().ndim() != 2 || num_heads <= 0 ||
        q.value().shape()[1] % static_cast<std::size_t>(num_heads) != 0) {
        throw std::invalid_argument("attention expects equal [tokens,channels] Q/K/V tensors");
    }
    const std::size_t tokens = q.value().shape()[0];
    const std::size_t channels = q.value().shape()[1];
    Tensor probabilities({static_cast<std::size_t>(num_heads), tokens, tokens},
                         q.value().backend());
    Tensor output(q.value().shape(), q.value().backend());
    q.value().backend()->causal_attention_forward(
        q.value().data(), k.value().data(), v.value().data(),
        probabilities.data(), output.data(), static_cast<int>(tokens),
        static_cast<int>(channels), num_heads);
    const bool track = q.requires_grad() || k.requires_grad() || v.requires_grad();
    return Variable::from_graph(std::move(output), track, {q, k, v},
        [q, k, v, num_heads, tokens, channels, probabilities](const Tensor& grad) {
            Tensor dq(q.value().shape(), q.value().backend());
            Tensor dk(k.value().shape(), k.value().backend());
            Tensor dv(v.value().shape(), v.value().backend());
            q.value().backend()->causal_attention_backward(
                q.value().data(), k.value().data(), v.value().data(), probabilities.data(),
                grad.data(), dq.data(), dk.data(), dv.data(), static_cast<int>(tokens),
                static_cast<int>(channels), num_heads);
            if (q.requires_grad()) q.accumulate_grad(dq);
            if (k.requires_grad()) k.accumulate_grad(dk);
            if (v.requires_grad()) v.accumulate_grad(dv);
        });
}

Variable cross_entropy(const Variable& logits, const std::vector<int>& targets,
                       const std::vector<float>& weights) {
    if (logits.value().ndim() != 2 || logits.value().shape()[0] != targets.size())
        throw std::invalid_argument("cross_entropy expects logits=[tokens,vocab] and one target per token");
    const std::size_t rows = logits.value().shape()[0];
    const std::size_t cols = logits.value().shape()[1];
    std::vector<float> target_values(rows);
    std::vector<float> weight_values = weights.empty() ? std::vector<float>(rows, 1.0f) : weights;
    if (weight_values.size() != rows)
        throw std::invalid_argument("cross_entropy expects one weight per token");
    float normalizer = 0.0f;
    for (std::size_t row = 0; row < rows; ++row) {
        if (targets[row] < 0 || static_cast<std::size_t>(targets[row]) >= cols)
            throw std::out_of_range("cross_entropy target is outside vocabulary");
        target_values[row] = static_cast<float>(targets[row]);
        if (!std::isfinite(weight_values[row]) || weight_values[row] < 0.0f)
            throw std::invalid_argument("cross_entropy weights must be finite and non-negative");
        normalizer += weight_values[row];
    }
    if (normalizer <= 0.0f)
        throw std::invalid_argument("cross_entropy requires at least one positive weight");
    Tensor target_tensor({rows}, logits.value().backend());
    target_tensor.copy_from(target_values);
    Tensor weight_tensor({rows}, logits.value().backend());
    weight_tensor.copy_from(weight_values);
    Tensor probabilities(logits.value().shape(), logits.value().backend());
    Tensor output({1}, logits.value().backend());
    logits.value().backend()->cross_entropy_forward(
        logits.value().data(), target_tensor.data(), weight_tensor.data(), normalizer,
        probabilities.data(), output.data(),
        static_cast<int>(rows), static_cast<int>(cols));
    return Variable::from_graph(std::move(output), logits.requires_grad(), {logits},
        [logits, target_tensor, weight_tensor, probabilities, normalizer, rows, cols](const Tensor& grad) {
            if (!logits.requires_grad()) return;
            Tensor dx(logits.value().shape(), logits.value().backend());
            logits.value().backend()->cross_entropy_backward(
                probabilities.data(), target_tensor.data(), weight_tensor.data(), normalizer,
                grad.data(), dx.data(),
                static_cast<int>(rows), static_cast<int>(cols));
            logits.accumulate_grad(dx);
        });
}

}  // namespace tinydl
