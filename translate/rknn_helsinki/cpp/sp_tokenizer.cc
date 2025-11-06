#include "sp_tokenizer.h"
#include <iostream>
#include <algorithm>
#include <fstream>

SPTokenizer::SPTokenizer() 
    : pad_token_id_(65000),
      eos_token_id_(0),
      unk_token_id_(3),
      model_loaded_(false) {
}

SPTokenizer::~SPTokenizer() {
}

bool SPTokenizer::load_model(const char* model_path, const char* vocab_path) {
    // 加载.spm
    auto status = processor_.Load(model_path);
    if (!status.ok()) return false;
    
    // 加载vocab.txt（改用txt）
    if (!load_vocab_txt(vocab_path)) return false;
    
    model_loaded_ = true;
    return true;
}

bool SPTokenizer::load_vocab_txt(const char* vocab_txt_path) {
    std::ifstream file(vocab_txt_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << vocab_txt_path << "\n";
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        
        std::string token = line.substr(0, tab);
        int64_t id = std::stoll(line.substr(tab + 1));
        
        vocab_[token] = id;
        id_to_piece_[id] = token;
    }
    
    file.close();
    std::cout << "[INFO] Loaded " << vocab_.size() << " tokens\n";
    return true;
}

std::vector<int64_t> SPTokenizer::encode(const std::string& text, bool add_eos) {
    if (!model_loaded_) {
        std::cerr << "[ERROR] Model not loaded!\n";
        return {};
    }
    
    // 1. SentencePiece分词
    std::vector<std::string> pieces;
    processor_.Encode(text, &pieces);
    
    // 2. 用vocab映射ID（关键修改）
    std::vector<int64_t> token_ids;
    for (const auto& piece : pieces) {
        auto it = vocab_.find(piece);  // ← 查vocab_映射表
        if (it != vocab_.end()) {
            token_ids.push_back(it->second);
            std::cout << "[DEBUG] '" << piece << "' -> " << it->second << "\n";
        } else {
            token_ids.push_back(unk_token_id_);
            std::cerr << "[WARN] Unknown piece: " << piece << "\n";
        }
    }
    
    if (add_eos) {
        token_ids.push_back(eos_token_id_);
    }
    
    return token_ids;
}

std::string SPTokenizer::decode(const std::vector<int64_t>& ids, bool skip_special) {
    if (!model_loaded_) {
        std::cerr << "[ERROR] Model not loaded!\n";
        return "";
    }
    
    std::vector<std::string> pieces;
    
    for (int64_t id : ids) {
        if (skip_special && (id == pad_token_id_ || id == eos_token_id_)) {
            continue;
        }
        
        auto it = id_to_piece_.find(id);
        if (it != id_to_piece_.end()) {
            pieces.push_back(it->second);
            // 调试输出
            std::cout << "[DECODE DEBUG] id=" << id << " -> '" << it->second << "'\n";
        } else {
            std::cout << "[DECODE WARN] id=" << id << " not found\n";
        }
    }
    
    // DecodePieces返回string，不是用指针
    std::string result = processor_.DecodePieces(pieces);  // ← 改这里
    return result;
}

std::vector<int64_t> SPTokenizer::pad_sequence(const std::vector<int64_t>& token_ids,
                                               int max_length, int64_t pad_token_id) {
    std::vector<int64_t> padded(max_length, pad_token_id);
    
    int copy_len = std::min((int)token_ids.size(), max_length);
    for (int i = 0; i < copy_len; i++) {
        padded[i] = token_ids[i];
    }
    
    return padded;
}

