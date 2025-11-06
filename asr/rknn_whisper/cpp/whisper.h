#ifndef _RKNN_DEMO_WHISPER_H_
#define _RKNN_DEMO_WHISPER_H_

#include "rknn_api.h"
#include "audio_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include "process.h"

// ==================== 零拷贝优化：更新结构体定义 ====================
typedef struct
{
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    
    // ✅ 新增：零拷贝内存管理
    rknn_tensor_mem **input_mems;   // 输入DMA内存
    rknn_tensor_mem **output_mems;  // 输出DMA内存
    bool use_zero_copy;              // 是否启用零拷贝
} rknn_app_context_t;

typedef struct
{
    rknn_app_context_t encoder_context;
    rknn_app_context_t decoder_context;
} rknn_whisper_context_t;

// ==================== 性能统计结构体 ====================
typedef struct {
    // Token统计
    int total_loops;           // 解码总循环次数
    int valid_tokens;          // 有效文本token数（排除时间戳、结束符）
    int timestamp_tokens;      // 时间戳token数
    
    // 时间统计（毫秒）
    double encoder_time_ms;    // 编码器耗时
    double decoder_time_ms;    // 解码器耗时
    double total_time_ms;      // 总耗时
} whisper_perf_stats_t;


int init_whisper_model(const char *model_path, rknn_app_context_t *app_ctx);
int release_whisper_model(rknn_app_context_t *app_ctx);
int inference_decoder_model(rknn_app_context_t *app_ctx, float *encoder_output,VocabEntry *vocab, int task_code,std::vector<std::string> &recognized_text,whisper_perf_stats_t *perf_stats);
int inference_whisper_model(rknn_whisper_context_t *app_ctx, std::vector<float> audio_data, float *mel_filters, VocabEntry *vocab, int task_code, std::vector<std::string> &recognized_text,whisper_perf_stats_t *perf_stats);

#endif //_RKNN_DEMO_WHISPER_H_