#include "tinydl/checkpoint.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

tinydl::Variable* parameter(tinydl::GptModel& model, const std::string& name) {
    for (auto& entry : model.named_parameters()) {
        if (entry.first == name) return entry.second;
    }
    throw std::runtime_error("missing parameter: " + name);
}

void expect_equal(const std::vector<float>& actual, const std::vector<float>& expected,
                  const std::string& message) {
    if (actual.size() != expected.size()) throw std::runtime_error(message + ": size mismatch");
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::fabs(actual[index] - expected[index]) > 1.0e-7f)
            throw std::runtime_error(message + ": value mismatch");
    }
}

void test_sequence_length_expansion(const std::shared_ptr<tinydl::Backend>& cpu) {
    tinydl::GptConfig source_config;
    source_config.vocab_size = 32;
    source_config.max_sequence_length = 4;
    source_config.channels = 8;
    source_config.num_heads = 2;
    source_config.num_layers = 1;
    source_config.ffn_channels = 16;
    tinydl::GptModel source(source_config, cpu);
    auto* source_positions = parameter(source, "position_embedding.weight");
    std::vector<float> source_values(source_positions->value().numel());
    for (std::size_t index = 0; index < source_values.size(); ++index)
        source_values[index] = static_cast<float>(index) / 100.0f;
    source_positions->value().copy_from(source_values);

    const auto path = std::filesystem::temp_directory_path() /
        "tinydl-checkpoint-sequence-expansion.tdl";
    tinydl::save_checkpoint(source, path.string());

    auto expanded_config = source_config;
    expanded_config.max_sequence_length = 8;
    tinydl::GptModel expanded(expanded_config, cpu);
    auto* expanded_positions = parameter(expanded, "position_embedding.weight");
    const auto initialized = expanded_positions->value().to_vector();
    tinydl::load_checkpoint(expanded, path.string(), true);
    const auto loaded = expanded_positions->value().to_vector();

    expect_equal(std::vector<float>(loaded.begin(), loaded.begin() + source_values.size()),
                 source_values, "copied position rows");
    expect_equal(std::vector<float>(loaded.begin() + source_values.size(), loaded.end()),
                 std::vector<float>(initialized.begin() + source_values.size(), initialized.end()),
                 "initialized position rows");
    if (expanded.forward({1, 2, 3, 4, 5, 6, 7, 8}).value().shape()[0] != 8)
        throw std::runtime_error("expanded model rejected the longer context");
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    try {
        test_sequence_length_expansion(tinydl::make_cpu_backend());
        std::cout << "All TinyDL checkpoint tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
