#pragma once

#include <string>
#include <vector>

namespace tinydl {

// A reversible byte-level tokenizer. UTF-8 text is treated as bytes, so the vocabulary is
// fixed at 256 and no external vocabulary file is needed.
class ByteTokenizer {
public:
    static constexpr int vocabulary_size = 256;

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& token_ids) const;
};

}  // namespace tinydl
