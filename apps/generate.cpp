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

int sample_last_token(const tinydl::Tensor& logits, float temperature, int top_k,
                      std::mt19937& random) {
    const auto values = logits.to_vector();
    const std::size_t vocab = logits.shape()[1];
    const std::size_t offset = (logits.shape()[0] - 1) * vocab;
    std::vector<int> indices(vocab);
    std::iota(indices.begin(), indices.end(), 0);
    top_k = std::max(1, std::min(top_k, static_cast<int>(vocab)));
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
    if (argc < 3) {
        std::cerr << "usage: tinydl_generate <checkpoint> <prompt> [tokens] [--dcu]\n";
        return 2;
    }
    try {
        const std::string checkpoint_path = argv[1];
        const std::string prompt = argv[2];
        int tokens_to_generate = 100;
        bool use_dcu = false;
        for (int i = 3; i < argc; ++i) {
            if (std::string(argv[i]) == "--dcu") use_dcu = true;
            else tokens_to_generate = std::stoi(argv[i]);
        }
        const auto config = tinydl::read_checkpoint_config(checkpoint_path);
        auto backend = use_dcu ? tinydl::make_dcu_backend() : tinydl::make_cpu_backend();
        tinydl::GptModel model(config, backend);
        tinydl::load_checkpoint(model, checkpoint_path);
        tinydl::ByteTokenizer tokenizer;
        auto tokens = tokenizer.encode(prompt);
        if (tokens.empty()) throw std::runtime_error("prompt must not be empty");
        std::mt19937 random(std::random_device{}());
        for (int step = 0; step < tokens_to_generate; ++step) {
            const std::size_t context_size = std::min(tokens.size(),
                static_cast<std::size_t>(config.max_sequence_length));
            std::vector<int> context(tokens.end() - static_cast<std::ptrdiff_t>(context_size), tokens.end());
            const int next = sample_last_token(model.forward(context).value(), 0.8f, 40, random);
            tokens.push_back(next);
        }
        std::cout << tokenizer.decode(tokens) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generation failed: " << error.what() << '\n';
        return 1;
    }
}
