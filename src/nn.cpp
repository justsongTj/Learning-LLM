#include "tinydl/nn.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace tinydl {
namespace {

Variable initialized_parameter(const std::vector<std::size_t>& shape,
                               const std::shared_ptr<Backend>& backend,
                               float standard_deviation) {
    static std::mt19937 generator(0x54494e59u);
    std::normal_distribution<float> distribution(0.0f, standard_deviation);
    Tensor tensor(shape, backend);
    std::vector<float> values(tensor.numel());
    for (float& value : values) value = distribution(generator);
    tensor.copy_from(values);
    return Variable(std::move(tensor), true);
}

void append(std::vector<NamedParameter>& destination, std::vector<NamedParameter> source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

}  // namespace

Linear::Linear(int input_features, int output_features, std::shared_ptr<Backend> backend)
    : weight_(initialized_parameter({static_cast<std::size_t>(input_features),
                                     static_cast<std::size_t>(output_features)},
                                    backend, 0.02f)),
      bias_(Tensor({static_cast<std::size_t>(output_features)}, backend), true) {
    bias_.value().fill(0.0f);
}

Variable Linear::forward(const Variable& input) const {
    return add_bias(tinydl::matmul(input, weight_), bias_);
}

std::vector<NamedParameter> Linear::named_parameters(const std::string& prefix) {
    return {{prefix + ".weight", &weight_}, {prefix + ".bias", &bias_}};
}

Embedding::Embedding(int count, int channels, std::shared_ptr<Backend> backend)
    : weight_(initialized_parameter({static_cast<std::size_t>(count),
                                     static_cast<std::size_t>(channels)},
                                    backend, 0.02f)) {}

Variable Embedding::forward(const std::vector<int>& indices) const {
    return tinydl::embedding(weight_, indices);
}

std::vector<NamedParameter> Embedding::named_parameters(const std::string& prefix) {
    return {{prefix + ".weight", &weight_}};
}

LayerNorm::LayerNorm(int channels, std::shared_ptr<Backend> backend, float eps)
    : weight_(Tensor({static_cast<std::size_t>(channels)}, backend), true),
      bias_(Tensor({static_cast<std::size_t>(channels)}, backend), true), eps_(eps) {
    weight_.value().fill(1.0f);
    bias_.value().fill(0.0f);
}

Variable LayerNorm::forward(const Variable& input) const {
    return tinydl::layer_norm(input, weight_, bias_, eps_);
}

std::vector<NamedParameter> LayerNorm::named_parameters(const std::string& prefix) {
    return {{prefix + ".weight", &weight_}, {prefix + ".bias", &bias_}};
}

CausalSelfAttention::CausalSelfAttention(const GptConfig& config,
                                         std::shared_ptr<Backend> backend)
    : num_heads_(config.num_heads),
      query_(config.channels, config.channels, backend),
      key_(config.channels, config.channels, backend),
      value_(config.channels, config.channels, backend),
      projection_(config.channels, config.channels, backend) {}

Variable CausalSelfAttention::forward(const Variable& input) const {
    auto context = causal_self_attention(query_.forward(input), key_.forward(input),
                                         value_.forward(input), num_heads_);
    return projection_.forward(context);
}

std::vector<NamedParameter> CausalSelfAttention::named_parameters(const std::string& prefix) {
    std::vector<NamedParameter> result;
    append(result, query_.named_parameters(prefix + ".query"));
    append(result, key_.named_parameters(prefix + ".key"));
    append(result, value_.named_parameters(prefix + ".value"));
    append(result, projection_.named_parameters(prefix + ".projection"));
    return result;
}

TransformerBlock::TransformerBlock(const GptConfig& config,
                                   std::shared_ptr<Backend> backend)
    : norm1_(config.channels, backend),
      attention_(config, backend),
      norm2_(config.channels, backend),
      feed_forward1_(config.channels, config.ffn_channels, backend),
      feed_forward2_(config.ffn_channels, config.channels, backend) {}

Variable TransformerBlock::forward(const Variable& input) const {
    auto after_attention = add(input, attention_.forward(norm1_.forward(input)));
    auto feed_forward = feed_forward2_.forward(gelu(feed_forward1_.forward(norm2_.forward(after_attention))));
    return add(after_attention, feed_forward);
}

std::vector<NamedParameter> TransformerBlock::named_parameters(const std::string& prefix) {
    std::vector<NamedParameter> result;
    append(result, norm1_.named_parameters(prefix + ".norm1"));
    append(result, attention_.named_parameters(prefix + ".attention"));
    append(result, norm2_.named_parameters(prefix + ".norm2"));
    append(result, feed_forward1_.named_parameters(prefix + ".ffn1"));
    append(result, feed_forward2_.named_parameters(prefix + ".ffn2"));
    return result;
}

GptModel::GptModel(GptConfig config, std::shared_ptr<Backend> backend)
    : config_(config), backend_(std::move(backend)),
      token_embedding_(config.vocab_size, config.channels, backend_),
      position_embedding_(config.max_sequence_length, config.channels, backend_),
      final_norm_(config.channels, backend_),
      language_model_head_(config.channels, config.vocab_size, backend_) {
    if (!backend_) throw std::invalid_argument("GptModel backend must not be null");
    if (config_.channels <= 0 || config_.num_heads <= 0 ||
        config_.channels % config_.num_heads != 0 || config_.num_layers <= 0)
        throw std::invalid_argument("invalid GPT configuration");
    blocks_.reserve(static_cast<std::size_t>(config_.num_layers));
    for (int i = 0; i < config_.num_layers; ++i)
        blocks_.push_back(std::make_unique<TransformerBlock>(config_, backend_));
}

Variable GptModel::forward(const std::vector<int>& token_ids) const {
    if (token_ids.empty() || token_ids.size() > static_cast<std::size_t>(config_.max_sequence_length))
        throw std::invalid_argument("token sequence length is outside model limits");
    std::vector<int> positions(token_ids.size());
    for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = static_cast<int>(i);
    auto hidden = add(token_embedding_.forward(token_ids), position_embedding_.forward(positions));
    for (const auto& block : blocks_) hidden = block->forward(hidden);
    return language_model_head_.forward(final_norm_.forward(hidden));
}

std::vector<NamedParameter> GptModel::named_parameters() {
    std::vector<NamedParameter> result;
    append(result, token_embedding_.named_parameters("token_embedding"));
    append(result, position_embedding_.named_parameters("position_embedding"));
    for (std::size_t i = 0; i < blocks_.size(); ++i)
        append(result, blocks_[i]->named_parameters("blocks." + std::to_string(i)));
    append(result, final_norm_.named_parameters("final_norm"));
    append(result, language_model_head_.named_parameters("lm_head"));
    return result;
}

void GptModel::zero_grad() {
    for (auto& entry : named_parameters()) entry.second->zero_grad();
}

}  // namespace tinydl
