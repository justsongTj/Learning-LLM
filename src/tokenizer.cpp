#include "tinydl/tokenizer.h"

#include <stdexcept>

namespace tinydl {

//将一个字符串编码成字节级别的token序列int类型的数组  中文字符可能是多字节的，所以这里是按字节来编码的  “你”-> 0xe4 0xbd 0xa0
std::vector<int> ByteTokenizer::encode(const std::string& text) const {
    std::vector<int> result;
    result.reserve(text.size());
    for (unsigned char byte : text) result.push_back(static_cast<int>(byte));
    return result;
}

//将字节级别的token序列解码成字符串
std::string ByteTokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    result.reserve(token_ids.size());
    for (int token : token_ids) {
        if (token < 0 || token >= vocabulary_size) throw std::out_of_range("invalid byte token");
        result.push_back(static_cast<char>(static_cast<unsigned char>(token)));
    }
    return result;
}

SentencePieceTokenizer::SentencePieceTokenizer(const std::string& model_path) {
    const auto status = processor_.Load(model_path);
    if (!status.ok())
        throw std::runtime_error("cannot load SentencePiece model: " + status.ToString());
}

int SentencePieceTokenizer::vocabulary_size() const {
    return processor_.GetPieceSize();
}

std::vector<int> SentencePieceTokenizer::encode(const std::string& text) const {
    std::vector<int> result;
    const auto status = processor_.Encode(text, &result);
    if (!status.ok()) throw std::runtime_error("SentencePiece encode failed: " + status.ToString());
    return result;
}

std::string SentencePieceTokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    const auto status = processor_.Decode(token_ids, &result);
    if (!status.ok()) throw std::runtime_error("SentencePiece decode failed: " + status.ToString());
    return result;
}

bool SentencePieceTokenizer::is_generatable(int token_id) const {
    return token_id == eos_id() ||
           (token_id >= 0 && token_id < vocabulary_size() &&
           !processor_.IsControl(token_id) && !processor_.IsUnknown(token_id) &&
           !processor_.IsUnused(token_id));
}

bool SentencePieceTokenizer::is_whitespace(int token_id) const {
    return token_id >= 0 && token_id < vocabulary_size() &&
           processor_.IdToPiece(token_id) == "▁";
}

int SentencePieceTokenizer::bos_id() const { return processor_.bos_id(); }
int SentencePieceTokenizer::eos_id() const { return processor_.eos_id(); }

}  // namespace tinydl
