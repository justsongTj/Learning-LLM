#include "tinydl/checkpoint.h"
#include "tinydl/optimizer.h"
#include "tinydl/tokenizer.h"

#include <algorithm>
#include <array>
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

std::vector<int> encode_corpus(const std::string& path,
                               const tinydl::SentencePieceTokenizer& tokenizer) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open training text: " + path);
    if (tokenizer.eos_id() < 0)
        throw std::runtime_error("SentencePiece model has no EOS token; retrain it with --eos_id=2");
    std::vector<int> result;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (tokenizer.bos_id() >= 0) result.push_back(tokenizer.bos_id());
        auto encoded = tokenizer.encode(line);
        result.insert(result.end(), encoded.begin(), encoded.end());
        result.push_back(tokenizer.eos_id());
    }
    if (!stream.eof()) throw std::runtime_error("failed while reading training text: " + path);
    return result;
}

std::shared_ptr<tinydl::Backend> select_backend(bool use_dcu) {
    return use_dcu ? tinydl::make_dcu_backend() : tinydl::make_cpu_backend();
}

float evaluate(tinydl::GptModel& model, const std::vector<int>& corpus,
               std::size_t begin, int sequence_length) {
    constexpr int max_samples = 32;
    float total = 0.0f;
    int samples = 0;
    for (std::size_t start = begin;
         start + static_cast<std::size_t>(sequence_length) < corpus.size() && samples < max_samples;
         start += static_cast<std::size_t>(sequence_length)) {
        std::vector<int> input(corpus.begin() + static_cast<std::ptrdiff_t>(start),
                               corpus.begin() + static_cast<std::ptrdiff_t>(start + sequence_length));
        std::vector<int> target(corpus.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                corpus.begin() + static_cast<std::ptrdiff_t>(start + sequence_length + 1));
        total += tinydl::cross_entropy(model.forward(input), target).value().to_vector()[0];
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
                     "[--validation-text path] "
                     "[--overfit-one-batch] [--dcu]\n";
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
        std::string resume_path;
        std::string validation_path;
        std::int64_t resume_step = -1;
        bool overfit_one_batch = false;
        bool use_dcu = false;
        for (int i = 4; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--dcu") use_dcu = true;
            else if (argument == "--overfit-one-batch") overfit_one_batch = true;
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
            else if (!argument.empty() && argument[0] != '-') steps = std::stoi(argument);
            else throw std::invalid_argument("unknown or incomplete option: " + argument);
        }
        if (steps <= 0) throw std::invalid_argument("steps must be positive");
        if (learning_rate <= 0.0f || min_learning_rate <= 0.0f || min_learning_rate > learning_rate)
            throw std::invalid_argument("learning rates must satisfy 0 < min <= initial");
        if (warmup_steps < 0 || validation_interval <= 0 || checkpoint_interval < 0)
            throw std::invalid_argument("warmup/checkpoint interval must be non-negative and validation interval positive");
        if (resume_step < -1) throw std::invalid_argument("resume step must be non-negative");
        if (sequence_length <= 0 || channels <= 0 || heads <= 0 || layers <= 0 ||
            ffn_channels <= 0 || channels % heads != 0)
            throw std::invalid_argument("model dimensions must be positive and channels divisible by heads");

        tinydl::SentencePieceTokenizer tokenizer(tokenizer_path);
        // Encode each paragraph independently and preserve its boundary with BOS/EOS.
        const auto corpus = encode_corpus(text_path, tokenizer);
        const auto external_validation = validation_path.empty()
            ? std::vector<int>{} : encode_corpus(validation_path, tokenizer);
        //定义要训练的GPT模型结构
        tinydl::GptConfig config;
        config.vocab_size = tokenizer.vocabulary_size();
        config.max_sequence_length = sequence_length;
        config.channels = channels;
        config.num_heads = heads;
        config.num_layers = layers;
        config.ffn_channels = ffn_channels;
        if (!resume_path.empty()) config = tinydl::read_checkpoint_config(resume_path);
        if (corpus.size() <= static_cast<std::size_t>(config.max_sequence_length))
            throw std::runtime_error("training text is shorter than one sequence");

        tinydl::GptModel model(config, select_backend(use_dcu));//创建模型，使用什么后端
        if (!resume_path.empty()) tinydl::load_checkpoint(model, resume_path);
        tinydl::AdamWOptions optimizer_options;
        optimizer_options.learning_rate = learning_rate;
        tinydl::AdamW optimizer(model.named_parameters(), optimizer_options);//创建AdamW优化器把模型所有可训练参数交给 AdamW，让 AdamW 负责在反向传播后更新它们。
        std::mt19937 random(42);//随机数生成器，用来随机选择训练语料片段的位置
        const std::size_t minimum_split = static_cast<std::size_t>(config.max_sequence_length + 1);
        const std::size_t validation_size = std::max(minimum_split, corpus.size() / 20);
        if (!overfit_one_batch && validation_path.empty() &&
            corpus.size() <= validation_size + minimum_split)
            throw std::runtime_error("training text is too short for a training/validation split");
        if (!overfit_one_batch && !validation_path.empty() &&
            external_validation.size() <= minimum_split)
            throw std::runtime_error("validation text is shorter than one sequence");
        const std::size_t training_end = overfit_one_batch || !validation_path.empty()
            ? corpus.size() : corpus.size() - validation_size;
        std::uniform_int_distribution<std::size_t> start_distribution(
            0, training_end - static_cast<std::size_t>(config.max_sequence_length) - 1);//定义随机起点范围 从tokenizer编码后的语料中随机选择一个起点，截取长度为max_sequence_length的片段作为训练样本

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
            const std::size_t start = overfit_one_batch ? 0 : start_distribution(random);
            //构造输入
            std::vector<int> input(corpus.begin() + static_cast<std::ptrdiff_t>(start),
                                   corpus.begin() + static_cast<std::ptrdiff_t>(start + config.max_sequence_length));
            //构造正确答案
            std::vector<int> target(corpus.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                    corpus.begin() + static_cast<std::ptrdiff_t>(start + config.max_sequence_length + 1));
            optimizer.zero_grad();
            auto loss = tinydl::cross_entropy(model.forward(input), target);
            const float loss_value = loss.value().to_vector()[0];
            loss.backward();
            optimizer.step();
            recent_losses.push_back(loss_value);
            recent_total += loss_value;
            if (recent_losses.size() > average_window) {
                recent_total -= recent_losses.front();
                recent_losses.pop_front();
            }
            if (step == 1 || step % 10 == 0) {
                std::cout << std::setprecision(6) << "step " << global_step
                          << " loss " << loss_value
                          << " avg_loss_" << recent_losses.size() << ' '
                          << recent_total / static_cast<float>(recent_losses.size())
                          << " lr " << current_learning_rate << '\n';
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
