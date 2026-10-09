#include "tinydl/tokenizer.h"

#include <stdexcept>

namespace tinydl {

std::vector<int> ByteTokenizer::encode(const std::string& text) const {
    std::vector<int> result;
    result.reserve(text.size());
    for (unsigned char byte : text) result.push_back(static_cast<int>(byte));
    return result;
}

std::string ByteTokenizer::decode(const std::vector<int>& token_ids) const {
    std::string result;
    result.reserve(token_ids.size());
    for (int token : token_ids) {
        if (token < 0 || token >= vocabulary_size) throw std::out_of_range("invalid byte token");
        result.push_back(static_cast<char>(static_cast<unsigned char>(token)));
    }
    return result;
}

}  // namespace tinydl
