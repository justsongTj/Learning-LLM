#include "tinydl/checkpoint.h"
#include "tinydl/optimizer.h"
#include "tinydl/tokenizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> training_magic{{'T', 'D', 'L', 'T', 'R', 'N', '0', '1'}};
constexpr std::array<char, 8> corpus_magic{{'T', 'D', 'L', 'C', 'R', 'P', '0', '1'}};

template <typename T>
void write_value(std::ostream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_value(std::istream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) throw std::runtime_error("truncated training state");
    return value;
}

void save_training_state(const std::string& model_path, std::uint64_t completed_steps,
                         const tinydl::AdamW& optimizer, const std::mt19937& random) {
    const std::string state_path = model_path + ".state";
    std::ofstream stream(state_path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot open training state for writing: " + state_path);
    stream.write(training_magic.data(), static_cast<std::streamsize>(training_magic.size()));
    write_value(stream, completed_steps);
    optimizer.save_state(stream);
    std::ostringstream random_stream;
    random_stream << random;
    const std::string random_state = random_stream.str();
    write_value(stream, static_cast<std::uint64_t>(random_state.size()));
    stream.write(random_state.data(), static_cast<std::streamsize>(random_state.size()));
    if (!stream) throw std::runtime_error("failed while writing training state: " + state_path);
}

std::uint64_t load_training_state(const std::string& model_path, tinydl::AdamW& optimizer,
                                  std::mt19937& random) {
    const std::string state_path = model_path + ".state";
    std::ifstream stream(state_path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open training state: " + state_path);
    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != training_magic) throw std::runtime_error("not a TinyDL training state");
    const std::uint64_t completed_steps = read_value<std::uint64_t>(stream);
    optimizer.load_state(stream);
    const auto random_size = read_value<std::uint64_t>(stream);
    std::string random_state(static_cast<std::size_t>(random_size), '\0');
    stream.read(random_state.data(), static_cast<std::streamsize>(random_state.size()));
    if (!stream) throw std::runtime_error("truncated random generator state");
    std::istringstream random_stream(random_state);
    random_stream >> random;
    if (!random_stream) throw std::runtime_error("invalid random generator state");
    return completed_steps;
}

struct EncodedCorpus {
    std::vector<int> tokens;
    std::vector<std::uint8_t> loss_weights;
    std::size_t instruction_paragraphs{0};
};

struct FileSignature {
    std::uint64_t size;
    std::int64_t modified;
};

FileSignature file_signature(const std::string& path) {
    const auto size = std::filesystem::file_size(path);
    const auto modified = std::filesystem::last_write_time(path).time_since_epoch().count();
    return {static_cast<std::uint64_t>(size), static_cast<std::int64_t>(modified)};
}

void save_corpus_cache(const std::string& cache_path, const std::string& text_path,
                       const std::string& tokenizer_path, bool answer_only_loss,
                       const EncodedCorpus& corpus) {
    static_assert(sizeof(int) == sizeof(std::int32_t));
    const std::string temporary_path = cache_path + ".part";
    std::ofstream stream(temporary_path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot open corpus cache for writing: " + temporary_path);
    const auto text = file_signature(text_path);
    const auto tokenizer = file_signature(tokenizer_path);
    stream.write(corpus_magic.data(), static_cast<std::streamsize>(corpus_magic.size()));
    write_value(stream, text.size);
    write_value(stream, text.modified);
    write_value(stream, tokenizer.size);
    write_value(stream, tokenizer.modified);
    write_value(stream, static_cast<std::uint8_t>(answer_only_loss));
    write_value(stream, static_cast<std::uint64_t>(corpus.instruction_paragraphs));
    write_value(stream, static_cast<std::uint64_t>(corpus.tokens.size()));
    stream.write(reinterpret_cast<const char*>(corpus.tokens.data()),
                 static_cast<std::streamsize>(corpus.tokens.size() * sizeof(int)));
    stream.write(reinterpret_cast<const char*>(corpus.loss_weights.data()),
                 static_cast<std::streamsize>(corpus.loss_weights.size()));
    if (!stream) throw std::runtime_error("failed while writing corpus cache: " + temporary_path);
    stream.close();
    std::error_code error;
    std::filesystem::remove(cache_path, error);
    error.clear();
    std::filesystem::rename(temporary_path, cache_path, error);
    if (error) throw std::runtime_error("cannot install corpus cache: " + error.message());
}

bool load_corpus_cache(const std::string& cache_path, const std::string& text_path,
                       const std::string& tokenizer_path, bool answer_only_loss,
                       EncodedCorpus& corpus) {
    std::ifstream stream(cache_path, std::ios::binary);
    if (!stream) return false;
    std::array<char, 8> magic{};
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || magic != corpus_magic) return false;
    const auto text_size = read_value<std::uint64_t>(stream);
    const auto text_modified = read_value<std::int64_t>(stream);
    const auto tokenizer_size = read_value<std::uint64_t>(stream);
    const auto tokenizer_modified = read_value<std::int64_t>(stream);
    const auto cached_answer_only = read_value<std::uint8_t>(stream);
    const auto instruction_paragraphs = read_value<std::uint64_t>(stream);
    const auto token_count = read_value<std::uint64_t>(stream);
    const auto text = file_signature(text_path);
    const auto tokenizer = file_signature(tokenizer_path);
    if (text_size != text.size || text_modified != text.modified ||
        tokenizer_size != tokenizer.size || tokenizer_modified != tokenizer.modified ||
        cached_answer_only != static_cast<std::uint8_t>(answer_only_loss))
        return false;
    const auto cache_size = std::filesystem::file_size(cache_path);
    constexpr std::uint64_t header_size = 8 + 8 + 8 + 8 + 8 + 1 + 8 + 8;
    const std::uint64_t expected_size = header_size + token_count * (sizeof(int) + 1);
    if (token_count == 0 || cache_size != expected_size) return false;
    corpus.tokens.resize(static_cast<std::size_t>(token_count));
    corpus.loss_weights.resize(static_cast<std::size_t>(token_count));
    corpus.instruction_paragraphs = static_cast<std::size_t>(instruction_paragraphs);
    stream.read(reinterpret_cast<char*>(corpus.tokens.data()),
                static_cast<std::streamsize>(corpus.tokens.size() * sizeof(int)));
    stream.read(reinterpret_cast<char*>(corpus.loss_weights.data()),
                static_cast<std::streamsize>(corpus.loss_weights.size()));
    return static_cast<bool>(stream);
}

EncodedCorpus encode_corpus(const std::string& path,
                            const tinydl::SentencePieceTokenizer& tokenizer,
                            bool answer_only_loss) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open training text: " + path);
    if (tokenizer.eos_id() < 0)
        throw std::runtime_error("SentencePiece model has no EOS token; retrain it with --eos_id=2");
    EncodedCorpus result;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const std::string marker = "回答:";
        const std::string full_width_marker = "回答：";
        std::size_t answer_position = answer_only_loss ? line.find(marker) : std::string::npos;
        std::size_t marker_size = marker.size();
        if (answer_only_loss && answer_position == std::string::npos) {
            answer_position = line.find(full_width_marker);
            marker_size = full_width_marker.size();
        }

        if (tokenizer.bos_id() >= 0) {
            result.tokens.push_back(tokenizer.bos_id());
            result.loss_weights.push_back(answer_position == std::string::npos ? 1.0f : 0.0f);
        }
        if (answer_position == std::string::npos) {
            auto encoded = tokenizer.encode(line);
            result.tokens.insert(result.tokens.end(), encoded.begin(), encoded.end());
            result.loss_weights.insert(result.loss_weights.end(), encoded.size(), 1.0f);
        } else {
            const std::size_t answer_begin = answer_position + marker_size;
            auto prefix = tokenizer.encode(line.substr(0, answer_begin));
            auto answer = tokenizer.encode(line.substr(answer_begin));
            result.tokens.insert(result.tokens.end(), prefix.begin(), prefix.end());
            result.loss_weights.insert(result.loss_weights.end(), prefix.size(), 0.0f);
            result.tokens.insert(result.tokens.end(), answer.begin(), answer.end());
            result.loss_weights.insert(result.loss_weights.end(), answer.size(), 1.0f);
            ++result.instruction_paragraphs;
        }
        result.tokens.push_back(tokenizer.eos_id());
        result.loss_weights.push_back(1.0f);
    }
    if (!stream.eof()) throw std::runtime_error("failed while reading training text: " + path);
    return result;
}

EncodedCorpus load_or_encode_corpus(const std::string& text_path,
                                    const std::string& tokenizer_path,
                                    const std::string& cache_path,
                                    const tinydl::SentencePieceTokenizer& tokenizer,
                                    bool answer_only_loss) {
    EncodedCorpus corpus;
    if (!cache_path.empty()) {
        try {
            if (load_corpus_cache(cache_path, text_path, tokenizer_path,
                                  answer_only_loss, corpus)) {
                std::cout << "loaded token cache " << cache_path << " ("
                          << corpus.tokens.size() << " tokens)\n";
                return corpus;
            }
            std::cout << "token cache is missing or stale: " << cache_path << '\n';
        } catch (const std::exception& error) {
            std::cout << "ignoring invalid token cache " << cache_path
                      << ": " << error.what() << '\n';
        }
    }
    corpus = encode_corpus(text_path, tokenizer, answer_only_loss);
    if (!cache_path.empty()) {
        save_corpus_cache(cache_path, text_path, tokenizer_path, answer_only_loss, corpus);
        std::cout << "saved token cache " << cache_path << " (" << corpus.tokens.size()
                  << " tokens)\n";
    }
    return corpus;
}

std::shared_ptr<tinydl::Backend> select_backend(bool use_dcu) {
    return use_dcu ? tinydl::make_dcu_backend() : tinydl::make_cpu_backend();
}

float evaluate(tinydl::GptModel& model, const EncodedCorpus& corpus,
               std::size_t begin, int sequence_length) {
    constexpr std::size_t max_samples = 32;
    const std::size_t length = static_cast<std::size_t>(sequence_length);
    if (begin + length >= corpus.tokens.size()) return 0.0f;

    const std::size_t available_starts = corpus.tokens.size() - begin - length;
    const std::size_t candidate_count = std::min(max_samples, available_starts);
    float total = 0.0f;
    std::size_t samples = 0;
    for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
        const std::size_t relative_start = candidate_count == 1
            ? 0 : candidate * (available_starts - 1) / (candidate_count - 1);
        const std::size_t start = begin + relative_start;
        std::vector<int> input(corpus.tokens.begin() + static_cast<std::ptrdiff_t>(start),
                               corpus.tokens.begin() + static_cast<std::ptrdiff_t>(start + length));
        std::vector<int> target(corpus.tokens.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                corpus.tokens.begin() + static_cast<std::ptrdiff_t>(start + length + 1));
        std::vector<float> weights(
            corpus.loss_weights.begin() + static_cast<std::ptrdiff_t>(start + 1),
            corpus.loss_weights.begin() + static_cast<std::ptrdiff_t>(start + length + 1));
        if (std::none_of(weights.begin(), weights.end(), [](float value) { return value > 0.0f; }))
            continue;
        total += tinydl::cross_entropy(model.forward(input), target, weights).value().to_vector()[0];
        ++samples;
    }
    return samples == 0 ? 0.0f : total / static_cast<float>(samples);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: tinydl_train <text-file> <tokenizer.model> <checkpoint> [steps] "
                     "[--learning-rate value] [--min-learning-rate value] [--warmup-steps value] "
                     "[--validation-interval value] [--sequence-length value] [--channels value] "
                     "[--heads value] [--layers value] [--ffn-channels value] "
                     "[--resume checkpoint] [--resume-step value] [--checkpoint-interval value] "
                     "[--validation-text path] [--gradient-accumulation-steps value] "
                     "[--training-token-cache path] [--validation-token-cache path] "
                     "[--answer-only-loss] [--overfit-one-batch] [--dcu]\n";
        return 2;
    }
    try {
        const std::string text_path = argv[1];
        const std::string tokenizer_path = argv[2];
        const std::string checkpoint_path = argv[3];
        int steps = 1000;
        float learning_rate = 3.0e-4f;
        float min_learning_rate = 3.0e-5f;
        int warmup_steps = 100;
        int validation_interval = 500;
        int sequence_length = 64;
        int channels = 128;
        int heads = 4;
        int layers = 4;
        int ffn_channels = 512;
        int checkpoint_interval = 5000;
        int gradient_accumulation_steps = 1;
        std::string resume_path;
        std::string validation_path;
        std::string training_cache_path;
        std::string validation_cache_path;
        std::int64_t resume_step = -1;
        bool overfit_one_batch = false;
        bool answer_only_loss = false;
        bool use_dcu = false;
        for (int i = 4; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--dcu") use_dcu = true;
            else if (argument == "--overfit-one-batch") overfit_one_batch = true;
            else if (argument == "--answer-only-loss") answer_only_loss = true;
            else if (argument == "--learning-rate" && i + 1 < argc) learning_rate = std::stof(argv[++i]);
            else if (argument == "--min-learning-rate" && i + 1 < argc) min_learning_rate = std::stof(argv[++i]);
            else if (argument == "--warmup-steps" && i + 1 < argc) warmup_steps = std::stoi(argv[++i]);
            else if (argument == "--validation-interval" && i + 1 < argc) validation_interval = std::stoi(argv[++i]);
            else if (argument == "--sequence-length" && i + 1 < argc) sequence_length = std::stoi(argv[++i]);
            else if (argument == "--channels" && i + 1 < argc) channels = std::stoi(argv[++i]);
            else if (argument == "--heads" && i + 1 < argc) heads = std::stoi(argv[++i]);
            else if (argument == "--layers" && i + 1 < argc) layers = std::stoi(argv[++i]);
            else if (argument == "--ffn-channels" && i + 1 < argc) ffn_channels = std::stoi(argv[++i]);
            else if (argument == "--resume" && i + 1 < argc) resume_path = argv[++i];
            else if (argument == "--resume-step" && i + 1 < argc) resume_step = std::stoll(argv[++i]);
            else if (argument == "--checkpoint-interval" && i + 1 < argc) checkpoint_interval = std::stoi(argv[++i]);
            else if (argument == "--validation-text" && i + 1 < argc) validation_path = argv[++i];
            else if (argument == "--training-token-cache" && i + 1 < argc)
                training_cache_path = argv[++i];
            else if (argument == "--validation-token-cache" && i + 1 < argc)
                validation_cache_path = argv[++i];
            else if (argument == "--gradient-accumulation-steps" && i + 1 < argc)
                gradient_accumulation_steps = std::stoi(argv[++i]);
            else if (!argument.empty() && argument[0] != '-') steps = std::stoi(argument);
            else throw std::invalid_argument("unknown or incomplete option: " + argument);
        }
        if (steps <= 0) throw std::invalid_argument("steps must be positive");
        if (learning_rate <= 0.0f || min_learning_rate <= 0.0f || min_learning_rate > learning_rate)
            throw std::invalid_argument("learning rates must satisfy 0 < min <= initial");
        if (warmup_steps < 0 || validation_interval <= 0 || checkpoint_interval < 0)
            throw std::invalid_argument("warmup/checkpoint interval must be non-negative and validation interval positive");
        if (resume_step < -1) throw std::invalid_argument("resume step must be non-negative");
        if (gradient_accumulation_steps <= 0)
            throw std::invalid_argument("gradient accumulation steps must be positive");
        if (!validation_cache_path.empty() && validation_path.empty())
            throw std::invalid_argument(
                "--validation-token-cache requires --validation-text");
        if (sequence_length <= 0 || channels <= 0 || heads <= 0 || layers <= 0 ||
            ffn_channels <= 0 || channels % heads != 0)
            throw std::invalid_argument("model dimensions must be positive and channels divisible by heads");

        tinydl::SentencePieceTokenizer tokenizer(tokenizer_path);
        // Encode each paragraph independently and preserve its boundary with BOS/EOS.
        const auto corpus = load_or_encode_corpus(
            text_path, tokenizer_path, training_cache_path, tokenizer, answer_only_loss);
        const auto external_validation = validation_path.empty()
            ? EncodedCorpus{} : load_or_encode_corpus(
                validation_path, tokenizer_path, validation_cache_path,
                tokenizer, answer_only_loss);
        if (answer_only_loss) {
            if (corpus.instruction_paragraphs == 0)
                throw std::runtime_error(
                    "--answer-only-loss found no paragraphs containing an answer marker");
            std::cout << "answer-only loss enabled for " << corpus.instruction_paragraphs
                      << " training paragraphs";
            if (!validation_path.empty())
                std::cout << " and " << external_validation.instruction_paragraphs
                          << " validation paragraphs";
            std::cout << '\n';
        }
        //定义要训练的GPT模型结构
        tinydl::GptConfig config;
        config.vocab_size = tokenizer.vocabulary_size();
        config.max_sequence_length = sequence_length;
        config.channels = channels;
        config.num_heads = heads;
        config.num_layers = layers;
        config.ffn_channels = ffn_channels;
        if (!resume_path.empty()) config = tinydl::read_checkpoint_config(resume_path);
        if (corpus.tokens.size() <= static_cast<std::size_t>(config.max_sequence_length))
            throw std::runtime_error("training text is shorter than one sequence");

        tinydl::GptModel model(config, select_backend(use_dcu));//创建模型，使用什么后端
        if (!resume_path.empty()) tinydl::load_checkpoint(model, resume_path);
        tinydl::AdamWOptions optimizer_options;
        optimizer_options.learning_rate = learning_rate;
        tinydl::AdamW optimizer(model.named_parameters(), optimizer_options);//创建AdamW优化器把模型所有可训练参数交给 AdamW，让 AdamW 负责在反向传播后更新它们。
        std::mt19937 random(42);//随机数生成器，用来随机选择训练语料片段的位置
        const std::size_t minimum_split = static_cast<std::size_t>(config.max_sequence_length + 1);
        const std::size_t validation_size = std::max(minimum_split, corpus.tokens.size() / 20);
        if (!overfit_one_batch && validation_path.empty() &&
            corpus.tokens.size() <= validation_size + minimum_split)
            throw std::runtime_error("training text is too short for a training/validation split");
        if (!overfit_one_batch && !validation_path.empty() &&
            external_validation.tokens.size() <= minimum_split)
            throw std::runtime_error("validation text is shorter than one sequence");
        const std::size_t training_end = overfit_one_batch || !validation_path.empty()
            ? corpus.tokens.size() : corpus.tokens.size() - validation_size;
        std::uniform_int_distribution<std::size_t> start_distribution(
            0, training_end - static_cast<std::size_t>(config.max_sequence_length) - 1);//定义随机起点范围 从tokenizer编码后的语料中随机选择一个起点，截取长度为max_sequence_length的片段作为训练样本
        std::size_t overfit_start = 0;
        if (overfit_one_batch && answer_only_loss) {
            const std::size_t length = static_cast<std::size_t>(config.max_sequence_length);
            while (overfit_start + length < training_end) {
                const auto weights_begin = corpus.loss_weights.begin() +
                    static_cast<std::ptrdiff_t>(overfit_start + 1);
                if (std::any_of(weights_begin, weights_begin + length,
                                [](float value) { return value > 0.0f; }))
                    break;
                ++overfit_start;
            }
            if (overfit_start + length >= training_end)
                throw std::runtime_error(
                    "could not find an overfit sequence with answer tokens");
        }

        std::uint64_t completed_steps = 0;
        if (!resume_path.empty()) {
            const std::string state_path = resume_path + ".state";
            if (std::filesystem::exists(state_path)) {
                completed_steps = load_training_state(resume_path, optimizer, random);
                if (resume_step >= 0 && static_cast<std::uint64_t>(resume_step) != completed_steps)
                    throw std::runtime_error("--resume-step does not match saved training state");
                std::cout << "resumed model and optimizer from step " << completed_steps << '\n';
            } else {
                if (resume_step < 0)
                    throw std::runtime_error("resume checkpoint has no .state file; provide --resume-step");
                completed_steps = static_cast<std::uint64_t>(resume_step);
                for (std::uint64_t i = 0; i < completed_steps; ++i)
                    static_cast<void>(start_distribution(random));
                std::cout << "resumed model weights from step " << completed_steps
                          << " with a fresh AdamW state\n";
            }
        }



        /*
        随机抽取一段文本
        → 构造输入和正确答案
        → GPT 前向传播
        → 计算 loss
        → 反向传播
        → AdamW 更新所有模型权重
        */
        std::deque<float> recent_losses;
        float recent_total = 0.0f;
        constexpr std::size_t average_window = 100;
        constexpr float pi = 3.14159265358979323846f;
        const int actual_warmup_steps = std::min(warmup_steps, std::max(0, steps / 10));
        auto throughput_start = std::chrono::steady_clock::now();
        int throughput_steps = 0;
        for (int step = 1; step <= steps; ++step)
        {
            const std::uint64_t global_step = completed_steps + static_cast<std::uint64_t>(step);
            float current_learning_rate = learning_rate;
            if (actual_warmup_steps > 0 && step <= actual_warmup_steps) {
                current_learning_rate = learning_rate * static_cast<float>(step) /
                                        static_cast<float>(actual_warmup_steps);
            } else if (steps > actual_warmup_steps) {
                const float progress = static_cast<float>(step - actual_warmup_steps) /
                                       static_cast<float>(steps - actual_warmup_steps);
                current_learning_rate = min_learning_rate + 0.5f * (learning_rate - min_learning_rate) *
                                        (1.0f + std::cos(pi * progress));
            }
            optimizer.set_learning_rate(current_learning_rate);
            optimizer.zero_grad();
            float loss_value = 0.0f;
            for (int micro_step = 0; micro_step < gradient_accumulation_steps; ++micro_step) {
                std::size_t start = overfit_one_batch ? overfit_start : start_distribution(random);
                if (answer_only_loss && !overfit_one_batch) {
                    for (int attempt = 0; attempt < 1024; ++attempt) {
                        const auto weights_begin = corpus.loss_weights.begin() +
                            static_cast<std::ptrdiff_t>(start + 1);
                        if (std::any_of(weights_begin,
                                        weights_begin + config.max_sequence_length,
                                        [](float value) { return value > 0.0f; }))
                            break;
                        start = start_distribution(random);
                        if (attempt == 1023)
                            throw std::runtime_error("could not sample a sequence with loss-bearing tokens");
                    }
                }
                std::vector<int> input(
                    corpus.tokens.begin() + static_cast<std::ptrdiff_t>(start),
                    corpus.tokens.begin() + static_cast<std::ptrdiff_t>(
                        start + config.max_sequence_length));
                std::vector<int> target(
                    corpus.tokens.begin() + static_cast<std::ptrdiff_t>(start + 1),
                    corpus.tokens.begin() + static_cast<std::ptrdiff_t>(
                        start + config.max_sequence_length + 1));
                std::vector<float> weights(
                    corpus.loss_weights.begin() + static_cast<std::ptrdiff_t>(start + 1),
                    corpus.loss_weights.begin() + static_cast<std::ptrdiff_t>(
                        start + config.max_sequence_length + 1));
                auto loss = tinydl::cross_entropy(model.forward(input), target, weights);
                loss_value += loss.value().to_vector()[0];
                loss.backward();
            }
            loss_value /= static_cast<float>(gradient_accumulation_steps);
            optimizer.step(1.0f / static_cast<float>(gradient_accumulation_steps));
            recent_losses.push_back(loss_value);
            ++throughput_steps;
            recent_total += loss_value;
            if (recent_losses.size() > average_window) {
                recent_total -= recent_losses.front();
                recent_losses.pop_front();
            }
            if (step == 1 || step % 10 == 0) {
                const auto now = std::chrono::steady_clock::now();
                const float elapsed = std::chrono::duration<float>(now - throughput_start).count();
                const float processed_tokens = static_cast<float>(
                    throughput_steps * gradient_accumulation_steps * config.max_sequence_length);
                std::cout << std::setprecision(6) << "step " << global_step
                          << " loss " << loss_value
                          << " avg_loss_" << recent_losses.size() << ' '
                          << recent_total / static_cast<float>(recent_losses.size())
                          << " lr " << current_learning_rate
                          << " tokens_per_second " << processed_tokens / elapsed << '\n';
                throughput_start = now;
                throughput_steps = 0;
            }
            if (!overfit_one_batch && (step % validation_interval == 0 || step == steps)) {
                const auto& validation_corpus = validation_path.empty() ? corpus : external_validation;
                const std::size_t validation_begin = validation_path.empty() ? training_end : 0;
                const float validation_loss = evaluate(model, validation_corpus, validation_begin,
                                                       config.max_sequence_length);
                std::cout << "step " << global_step << " validation_loss " << validation_loss
                          << " perplexity " << std::exp(validation_loss) << '\n';
            }
            if (checkpoint_interval > 0 && step % checkpoint_interval == 0 && step != steps) {
                const std::string periodic_path = checkpoint_path + ".step" +
                                                  std::to_string(global_step) + ".tdl";
                tinydl::save_checkpoint(model, periodic_path);
                save_training_state(periodic_path, global_step, optimizer, random);
                std::cout << "saved periodic checkpoint to " << periodic_path << '\n';
            }
        }
        tinydl::save_checkpoint(model, checkpoint_path);
        save_training_state(checkpoint_path, completed_steps + static_cast<std::uint64_t>(steps),
                            optimizer, random);
        std::cout << "saved checkpoint to " << checkpoint_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "training failed: " << error.what() << '\n';
        return 1;
    }
}
