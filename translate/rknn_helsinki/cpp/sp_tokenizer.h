#ifndef SP_TOKENIZER_H
#define SP_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include "sentencepiece_processor.h"

class SPTokenizer {
public:
    SPTokenizer();
    ~SPTokenizer();

    bool load_model(const char* model_path, const char* vocab_path);  // ← 加vocab参数
    std::vector<int64_t> encode(const std::string& text, bool add_eos = true);
    std::string decode(const std::vector<int64_t>& ids, bool skip_special = true);
    std::vector<int64_t> pad_sequence(const std::vector<int64_t>& ids, int max_len, int64_t pad_id);
    
    int64_t get_pad_token_id() const { return pad_token_id_; }
    int64_t get_eos_token_id() const { return eos_token_id_; }
    int64_t get_unk_token_id() const { return unk_token_id_; }
    

private:
    bool load_vocab_txt(const char* vocab_path);   // ← 新增
    
    sentencepiece::SentencePieceProcessor processor_;
    std::unordered_map<std::string, int64_t> vocab_;  // ← piece -> real_id
    std::unordered_map<int64_t, std::string> id_to_piece_;  // ← real_id -> piece
    
    int64_t pad_token_id_;
    int64_t eos_token_id_;
    int64_t unk_token_id_;
    bool model_loaded_;
};

#endif