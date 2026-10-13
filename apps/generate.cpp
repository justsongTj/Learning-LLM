#include "tinydl/checkpoint.h"
#include "tinydl/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int sample_last_token(const tinydl::Tensor& logits,
                      const tinydl::SentencePieceTokenizer& tokenizer,
                      float temperature, int top_k, std::mt19937& random) {
    const auto values = logits.to_vector();
    const std::size_t vocab = logits.shape()[1];
    const std::size_t offset = (logits.shape()[0] - 1) * vocab;
    std::vector<int> indices;
    indices.reserve(vocab);
    for (std::size_t i = 0; i < vocab; ++i)
        if (tokenizer.is_generatable(static_cast<int>(i))) indices.push_back(static_cast<int>(i));
    if (indices.empty()) throw std::runtime_error("SentencePiece vocabulary has no generatable tokens");
    top_k = std::max(1, std::min(top_k, static_cast<int>(indices.size())));
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [&](int a, int b) { return values[offset + static_cast<std::size_t>(a)] >
                                   values[offset + static_cast<std::size_t>(b)]; });
    float maximum = values[offset + static_cast<std::size_t>(indices[0])] / temperature;
    std::vector<double> weights(static_cast<std::size_t>(top_k));
    for (int i = 0; i < top_k; ++i)
        weights[static_cast<std::size_t>(i)] = std::exp(
            values[offset + static_cast<std::size_t>(indices[static_cast<std::size_t>(i)])] /
            temperature - maximum);
    std::discrete_distribution<int> distribution(weights.begin(), weights.end());
    return indices[static_cast<std::size_t>(distribution(random))];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: tinydl_generate <checkpoint> <tokenizer.model> <prompt> [tokens] "
                     "[--temperature value] [--top-k value] [--seed value] [--dcu]\n";
        return 2;
    }
    try {
        const std::string checkpoint_path = argv[1];
        const std::string tokenizer_path = argv[2];
        const std::string prompt = argv[3];
        int tokens_to_generate = 100;
        float temperature = 0.8f;
        int top_k = 40;
        unsigned int seed = std::random_device{}();
        bool use_dcu = false;
        for (int i = 4; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--dcu") use_dcu = true;
            else if (argument == "--temperature" && i + 1 < argc) temperature = std::stof(argv[++i]);
            else if (argument == "--top-k" && i + 1 < argc) top_k = std::stoi(argv[++i]);
            else if (argument == "--seed" && i + 1 < argc) seed = static_cast<unsigned int>(std::stoul(argv[++i]));
            else if (!argument.empty() && argument[0] != '-') tokens_to_generate = std::stoi(argument);
            else throw std::invalid_argument("unknown or incomplete option: " + argument);
        }
        if (tokens_to_generate < 0) throw std::invalid_argument("tokens must not be negative");
        if (temperature <= 0.0f) throw std::invalid_argument("temperature must be positive");
        if (top_k <= 0) throw std::invalid_argument("top-k must be positive");
        const auto config = tinydl::read_checkpoint_config(checkpoint_path);
        auto backend = use_dcu ? tinydl::make_dcu_backend() : tinydl::make_cpu_backend();
        tinydl::GptModel model(config, backend);
        tinydl::load_checkpoint(model, checkpoint_path);
        tinydl::SentencePieceTokenizer tokenizer(tokenizer_path);
        if (tokenizer.vocabulary_size() != config.vocab_size)
            throw std::runtime_error("SentencePiece vocabulary does not match checkpoint");
        auto tokens = tokenizer.encode(prompt);
        if (tokens.empty()) throw std::runtime_error("prompt must not be empty");
        std::mt19937 random(seed);
        for (int step = 0; step < tokens_to_generate; ++step) {
            const std::size_t context_size = std::min(tokens.size(),
                static_cast<std::size_t>(config.max_sequence_length));
            std::vector<int> context(tokens.end() - static_cast<std::ptrdiff_t>(context_size), tokens.end());
            const int next = sample_last_token(model.forward(context).value(), tokenizer,
                                               temperature, top_k, random);
            tokens.push_back(next);
        }
        std::cout << tokenizer.decode(tokens) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generation failed: " << error.what() << '\n';
        return 1;
    }
}
