#include "tinydl/checkpoint.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace tinydl {
namespace {

constexpr std::array<char, 8> magic{{'T', 'I', 'N', 'Y', 'D', 'L', '0', '1'}};

template <typename T>
void write_value(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_value(std::istream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) throw std::runtime_error("truncated TinyDL checkpoint");
    return value;
}

void write_config(std::ostream& stream, const GptConfig& config) {
    write_value(stream, config.vocab_size);
    write_value(stream, config.max_sequence_length);
    write_value(stream, config.channels);
    write_value(stream, config.num_heads);
    write_value(stream, config.num_layers);
    write_value(stream, config.ffn_channels);
}

GptConfig read_config(std::istream& stream) {
    GptConfig config;
    config.vocab_size = read_value<int>(stream);
    config.max_sequence_length = read_value<int>(stream);
    config.channels = read_value<int>(stream);
    config.num_heads = read_value<int>(stream);
    config.num_layers = read_value<int>(stream);
    config.ffn_channels = read_value<int>(stream);
    return config;
}

void require_magic(std::istream& stream) {
    std::array<char, 8> found{};
    stream.read(found.data(), static_cast<std::streamsize>(found.size()));
    if (!stream || found != magic) throw std::runtime_error("not a TinyDL checkpoint");
}

}  // namespace

void save_checkpoint(GptModel& model, const std::string& path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot open checkpoint for writing: " + path);
    stream.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_config(stream, model.config());
    const auto parameters = model.named_parameters();
    write_value(stream, static_cast<std::uint32_t>(parameters.size()));
    for (const auto& entry : parameters) {
        write_value(stream, static_cast<std::uint32_t>(entry.first.size()));
        stream.write(entry.first.data(), static_cast<std::streamsize>(entry.first.size()));
        const auto& shape = entry.second->value().shape();
        write_value(stream, static_cast<std::uint32_t>(shape.size()));
        for (std::size_t dimension : shape) write_value(stream, static_cast<std::uint64_t>(dimension));
        const auto values = entry.second->value().to_vector();
        stream.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    if (!stream) throw std::runtime_error("failed while writing checkpoint: " + path);
}

void load_checkpoint(GptModel& model, const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open checkpoint: " + path);
    require_magic(stream);
    const GptConfig stored = read_config(stream);
    const GptConfig& expected = model.config();
    if (stored.vocab_size != expected.vocab_size ||
        stored.max_sequence_length != expected.max_sequence_length ||
        stored.channels != expected.channels || stored.num_heads != expected.num_heads ||
        stored.num_layers != expected.num_layers || stored.ffn_channels != expected.ffn_channels)
        throw std::runtime_error("checkpoint model configuration does not match");

    std::unordered_map<std::string, Variable*> parameters;
    for (auto& entry : model.named_parameters()) parameters.emplace(entry.first, entry.second);
    const auto count = read_value<std::uint32_t>(stream);
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto name_size = read_value<std::uint32_t>(stream);
        std::string name(name_size, '\0');
        stream.read(name.data(), static_cast<std::streamsize>(name_size));
        const auto rank = read_value<std::uint32_t>(stream);
        std::vector<std::size_t> shape(rank);
        std::size_t elements = 1;
        for (auto& dimension : shape) {
            dimension = static_cast<std::size_t>(read_value<std::uint64_t>(stream));
            elements *= dimension;
        }
        std::vector<float> values(elements);
        stream.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!stream) throw std::runtime_error("truncated parameter: " + name);
        const auto found = parameters.find(name);
        if (found == parameters.end()) throw std::runtime_error("unknown checkpoint parameter: " + name);
        if (found->second->value().shape() != shape) throw std::runtime_error("shape mismatch: " + name);
        found->second->value().copy_from(values);
        parameters.erase(found);
    }
    if (!parameters.empty()) throw std::runtime_error("checkpoint is missing model parameters");
}

GptConfig read_checkpoint_config(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open checkpoint: " + path);
    require_magic(stream);
    return read_config(stream);
}

}  // namespace tinydl
