#include "tinydl/checkpoint.h"
#include "tinydl/optimizer.h"
#include "tinydl/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
            else if (!argument.empty() && argument[0] != '-') steps = std::stoi(argument);
            else throw std::invalid_argument("unknown or incomplete option: " + argument);
        }
        if (steps <= 0) throw std::invalid_argument("steps must be positive");
        if (learning_rate <= 0.0f || min_learning_rate <= 0.0f || min_learning_rate > learning_rate)
            throw std::invalid_argument("learning rates must satisfy 0 < min <= initial");
        if (warmup_steps < 0 || validation_interval <= 0)
            throw std::invalid_argument("warmup must be non-negative and validation interval positive");
        if (sequence_length <= 0 || channels <= 0 || heads <= 0 || layers <= 0 ||
            ffn_channels <= 0 || channels % heads != 0)
            throw std::invalid_argument("model dimensions must be positive and channels divisible by heads");

        tinydl::SentencePieceTokenizer tokenizer(tokenizer_path);
        // Encode each paragraph independently and preserve its boundary with BOS/EOS.
        const auto corpus = encode_corpus(text_path, tokenizer);
        //定义要训练的GPT模型结构
        tinydl::GptConfig config;
        config.vocab_size = tokenizer.vocabulary_size();
        config.max_sequence_length = sequence_length;
        config.channels = channels;
        config.num_heads = heads;
        config.num_layers = layers;
        config.ffn_channels = ffn_channels;
        if (corpus.size() <= static_cast<std::size_t>(config.max_sequence_length))
            throw std::runtime_error("training text is shorter than one sequence");

        tinydl::GptModel model(config, select_backend(use_dcu));//创建模型，使用什么后端
        tinydl::AdamWOptions optimizer_options;
        optimizer_options.learning_rate = learning_rate;
        tinydl::AdamW optimizer(model.named_parameters(), optimizer_options);//创建AdamW优化器把模型所有可训练参数交给 AdamW，让 AdamW 负责在反向传播后更新它们。
        std::mt19937 random(42);//随机数生成器，用来随机选择训练语料片段的位置
        const std::size_t minimum_split = static_cast<std::size_t>(config.max_sequence_length + 1);
        const std::size_t validation_size = std::max(minimum_split, corpus.size() / 20);
        if (!overfit_one_batch && corpus.size() <= validation_size + minimum_split)
            throw std::runtime_error("training text is too short for a training/validation split");
        const std::size_t training_end = overfit_one_batch ? corpus.size() : corpus.size() - validation_size;
        std::uniform_int_distribution<std::size_t> start_distribution(
            0, training_end - static_cast<std::size_t>(config.max_sequence_length) - 1);//定义随机起点范围 从tokenizer编码后的语料中随机选择一个起点，截取长度为max_sequence_length的片段作为训练样本



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
                std::cout << std::setprecision(6) << "step " << step
                          << " loss " << loss_value
                          << " avg_loss_" << recent_losses.size() << ' '
                          << recent_total / static_cast<float>(recent_losses.size())
                          << " lr " << current_learning_rate << '\n';
            }
            if (!overfit_one_batch && (step % validation_interval == 0 || step == steps)) {
                const float validation_loss = evaluate(model, corpus, training_end,
                                                       config.max_sequence_length);
                std::cout << "step " << step << " validation_loss " << validation_loss
                          << " perplexity " << std::exp(validation_loss) << '\n';
            }
        }
        tinydl::save_checkpoint(model, checkpoint_path);
        std::cout << "saved checkpoint to " << checkpoint_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "training failed: " << error.what() << '\n';
        return 1;
    }
}
