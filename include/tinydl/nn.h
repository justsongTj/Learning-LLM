#pragma once

#include "tinydl/autograd.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tinydl {

using NamedParameter = std::pair<std::string, Variable*>;

struct GptConfig {
    int vocab_size{256};
    int max_sequence_length{128};
    int channels{128};
    int num_heads{4};
    int num_layers{4};
    int ffn_channels{512};
};

class Linear {
public:
    Linear(int input_features, int output_features, std::shared_ptr<Backend> backend);
    Variable forward(const Variable& input) const;
    std::vector<NamedParameter> named_parameters(const std::string& prefix);

private:
    Variable weight_;
    Variable bias_;
};

class Embedding {
public:
    Embedding(int count, int channels, std::shared_ptr<Backend> backend);
    Variable forward(const std::vector<int>& indices) const;
    std::vector<NamedParameter> named_parameters(const std::string& prefix);

private:
    Variable weight_;
};

class LayerNorm {
public:
    LayerNorm(int channels, std::shared_ptr<Backend> backend, float eps = 1.0e-5f);
    Variable forward(const Variable& input) const;
    std::vector<NamedParameter> named_parameters(const std::string& prefix);

private:
    Variable weight_;
    Variable bias_;
    float eps_;
};

class CausalSelfAttention {
public:
    CausalSelfAttention(const GptConfig& config, std::shared_ptr<Backend> backend);
    Variable forward(const Variable& input) const;
    std::vector<NamedParameter> named_parameters(const std::string& prefix);

private:
    int num_heads_;
    Linear query_;
    Linear key_;
    Linear value_;
    Linear projection_;
};

class TransformerBlock {
public:
    TransformerBlock(const GptConfig& config, std::shared_ptr<Backend> backend);
    Variable forward(const Variable& input) const;
    std::vector<NamedParameter> named_parameters(const std::string& prefix);

private:
    LayerNorm norm1_;
    CausalSelfAttention attention_;
    LayerNorm norm2_;
    Linear feed_forward1_;
    Linear feed_forward2_;
};

class GptModel {
public:
    GptModel(GptConfig config, std::shared_ptr<Backend> backend);

    Variable forward(const std::vector<int>& token_ids) const;
    std::vector<NamedParameter> named_parameters();
    const GptConfig& config() const noexcept { return config_; }
    const std::shared_ptr<Backend>& backend() const noexcept { return backend_; }
    void zero_grad();

private:
    GptConfig config_;
    std::shared_ptr<Backend> backend_;
    Embedding token_embedding_;
    Embedding position_embedding_;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    LayerNorm final_norm_;
    Linear language_model_head_;
};

}  // namespace tinydl
