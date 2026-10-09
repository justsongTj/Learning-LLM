#include "tinydl/checkpoint.h"
#include "tinydl/optimizer.h"
#include "tinydl/tokenizer.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string read_text(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open training text: " + path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::shared_ptr<tinydl::Backend> select_backend(bool use_dcu) {
    return use_dcu ? tinydl::make_dcu_backend() : tinydl::make_cpu_backend();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: tinydl_train <text-file> <checkpoint> [steps] [--dcu]\n";
        return 2;
    }
    try {
        const std::string text_path = argv[1];
        const std::string checkpoint_path = argv[2];
        int steps = 1000;
        bool use_dcu = false;
        for (int i = 3; i < argc; ++i) {
            if (std::string(argv[i]) == "--dcu") use_dcu = true;
            else steps = std::stoi(argv[i]);
        }

        tinydl::ByteTokenizer tokenizer;
        const auto corpus = tokenizer.encode(read_text(text_path));
        tinydl::GptConfig config;
        config.vocab_size = tinydl::ByteTokenizer::vocabulary_size;
        config.max_sequence_length = 64;
        config.channels = 128;
        config.num_heads = 4;
        config.num_layers = 4;
        config.ffn_channels = 512;
        if (corpus.size() <= static_cast<std::size_t>(config.max_sequence_length))
            throw std::runtime_error("training text is shorter than one sequence");

        tinydl::GptModel model(config, select_backend(use_dcu));
        tinydl::AdamW optimizer(model.named_parameters());
        std::mt19937 random(42);
        std::uniform_int_distribution<std::size_t> start_distribution(
            0, corpus.size() - static_cast<std::size_t>(config.max_sequence_length) - 1);

        for (int step = 1; step <= steps; ++step) {
            const std::size_t start = start_distribution(random);
            std::vector<int> input(corpus.begin() + static_cast<std::ptrdiff_t>(start),
                                   corpus.begin() + static_cast<std::ptrdiff_t>(start + config.max_sequence_length));
            std::vector<int> target(corpus.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                    corpus.begin() + static_cast<std::ptrdiff_t>(start + config.max_sequence_length + 1));
            optimizer.zero_grad();
            auto loss = tinydl::cross_entropy(model.forward(input), target);
            const float loss_value = loss.value().to_vector()[0];
            loss.backward();
            optimizer.step();
            if (step == 1 || step % 10 == 0)
                std::cout << "step " << step << " loss " << loss_value << '\n';
        }
        tinydl::save_checkpoint(model, checkpoint_path);
        std::cout << "saved checkpoint to " << checkpoint_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "training failed: " << error.what() << '\n';
        return 1;
    }
}
