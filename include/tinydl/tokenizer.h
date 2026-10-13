#pragma once

#include <string>
#include <vector>

#include <sentencepiece_processor.h>

namespace tinydl {

// A reversible byte-level tokenizer. UTF-8 text is treated as bytes, so the vocabulary is
// fixed at 256 and no external vocabulary file is needed.
class ByteTokenizer {
public:
    static constexpr int vocabulary_size = 256;

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& token_ids) const;
};

// A subword tokenizer backed by a SentencePiece .model file. The same model must be used
// for training and generation because its token IDs define the GPT vocabulary.
class SentencePieceTokenizer {
public:
    explicit SentencePieceTokenizer(const std::string& model_path);

    int vocabulary_size() const;
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& token_ids) const;
    bool is_generatable(int token_id) const;

private:
    sentencepiece::SentencePieceProcessor processor_;
};

}  // namespace tinydl
