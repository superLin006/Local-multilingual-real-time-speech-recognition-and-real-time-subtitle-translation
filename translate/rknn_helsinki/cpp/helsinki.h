#ifndef HELSINKI_H
#define HELSINKI_H

#include <vector>
#include "rknn_api.h"

// ==================== 性能统计结构体 ====================
typedef struct {
    // 时间统计（毫秒）
    double encoder_time_ms;        // 编码器耗时
    double decoder_time_ms;        // 解码器总耗时
    double decoder_avg_time_ms;    // 解码器平均每步耗时
    double total_time_ms;          // 总耗时
    
    // Token统计
    int total_steps;               // 解码总步数
    int output_tokens;             // 输出token数
    
    // 内存统计
    bool zero_copy_enabled;        // 是否启用零拷贝
    size_t encoder_dma_size;       // Encoder DMA内存大小
    size_t decoder_dma_size;       // Decoder DMA内存大小
} helsinki_perf_stats_t;

class HelsinkiTranslator {
public:
    HelsinkiTranslator();
    ~HelsinkiTranslator();

    int init(const char* encoder_path, const char* decoder_path);
    int translate(const int64_t* input_ids, const int64_t* attention_mask,
                  int seq_len, int64_t* output_ids, int max_output_len,
                  helsinki_perf_stats_t* perf_stats = nullptr);
    
    void apply_repetition_penalty(float* logits, int vocab_size,
                                 const std::vector<int64_t>& generated_tokens,
                                 float penalty);
    void block_repeated_ngrams(float* logits, int vocab_size,
                              const std::vector<int64_t>& generated_tokens,
                              int ngram_size);
    void print_model_info();
    void release();

private:
    // ==================== 推理函数 ====================
    int run_encoder(const int64_t* input_ids, const int64_t* attention_mask,
                    int input_len, float* encoder_output);
    int run_encoder_zerocopy(const int64_t* input_ids, const int64_t* attention_mask,
                            int input_len, float* encoder_output);
    
    int run_decoder(const int64_t* decoder_input_ids, int decoder_input_len,
                    const float* encoder_hidden_states,
                    const int64_t* encoder_attention_mask,
                    float* logits);
    int run_decoder_zerocopy(const int64_t* decoder_input_ids, int decoder_input_len,
                            const float* encoder_hidden_states,
                            const int64_t* encoder_attention_mask,
                            float* logits);
    
    int argmax(const float* logits, int size);

    // ==================== Zero-Copy 辅助函数 ====================
    void float_to_fp16_neon(const float* src, uint16_t* dst, size_t count);
    void fp16_to_float_neon(const uint16_t* src, float* dst, size_t count);
    int init_zero_copy_memory();
    void release_zero_copy_memory();

    // ==================== 基础成员 ====================
    rknn_context encoder_ctx_;
    rknn_context decoder_ctx_;
    
    int encoder_seq_len_;
    int decoder_seq_len_;
    int hidden_size_;
    int vocab_size_;
    int pad_token_id_;
    int eos_token_id_;
    int decoder_start_token_id_;
    bool initialized_;

    // ==================== Zero-Copy 成员 ====================
    bool use_zero_copy_;
    
    // Encoder
    rknn_tensor_mem** encoder_input_mems_;    // 2个输入：input_ids, attention_mask
    rknn_tensor_mem** encoder_output_mems_;   // 1个输出：hidden_states
    rknn_tensor_attr* encoder_input_attrs_;
    rknn_tensor_attr* encoder_output_attrs_;
    int encoder_input_num_;
    int encoder_output_num_;
    
    // Decoder
    rknn_tensor_mem** decoder_input_mems_;    // 3个输入：input_ids, encoder_hidden_states, encoder_attention_mask
    rknn_tensor_mem** decoder_output_mems_;   // 1个输出：logits
    rknn_tensor_attr* decoder_input_attrs_;
    rknn_tensor_attr* decoder_output_attrs_;
    int decoder_input_num_;
    int decoder_output_num_;
};

#endif // HELSINKI_H
