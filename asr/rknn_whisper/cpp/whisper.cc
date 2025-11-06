#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "whisper.h"
#include "file_utils.h"
#include "audio_utils.h"
#include <vector>
#include "process.h"
#include <chrono>
#include <iomanip>
#include <arm_neon.h>

// ==================== FP16转换函数 (NEON优化) ====================

// Float转FP16 (使用ARM NEON硬件加速)
static void float_to_fp16_neon(const float* src, uint16_t* dst, size_t count) {
    size_t i = 0;
    
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // 使用FP16向量指令 (如果CPU支持)
    for (; i + 7 < count; i += 8) {
        float32x4_t v1 = vld1q_f32(src + i);
        float32x4_t v2 = vld1q_f32(src + i + 4);
        
        float16x4_t h1 = vcvt_f16_f32(v1);
        float16x4_t h2 = vcvt_f16_f32(v2);
        
        vst1_u16(dst + i, vreinterpret_u16_f16(h1));
        vst1_u16(dst + i + 4, vreinterpret_u16_f16(h2));
    }
#endif
    
    // 每次处理4个float
    for (; i + 3 < count; i += 4) {
        float32x4_t vf = vld1q_f32(src + i);
        float16x4_t vh = vcvt_f16_f32(vf);
        vst1_u16(dst + i, vreinterpret_u16_f16(vh));
    }
    
    // 处理剩余元素
    for (; i < count; i++) {
        // IEEE 754 float32转fp16的简化版本
        uint32_t f = *((uint32_t*)&src[i]);
        uint32_t sign = (f >> 16) & 0x8000;
        int32_t exp = ((f >> 23) & 0xff) - 127 + 15;
        uint32_t frac = (f >> 13) & 0x3ff;
        
        if (exp <= 0) {
            dst[i] = sign;  // 下溢为0
        } else if (exp >= 31) {
            dst[i] = sign | 0x7c00;  // 上溢为inf
        } else {
            dst[i] = sign | (exp << 10) | frac;
        }
    }
}

// FP16转Float (使用ARM NEON硬件加速)
static void fp16_to_float_neon(const uint16_t* src, float* dst, size_t count) {
    size_t i = 0;
    
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // 使用FP16向量指令
    for (; i + 7 < count; i += 8) {
        float16x4_t h1 = vreinterpret_f16_u16(vld1_u16(src + i));
        float16x4_t h2 = vreinterpret_f16_u16(vld1_u16(src + i + 4));
        
        float32x4_t v1 = vcvt_f32_f16(h1);
        float32x4_t v2 = vcvt_f32_f16(h2);
        
        vst1q_f32(dst + i, v1);
        vst1q_f32(dst + i + 4, v2);
    }
#endif
    
    // 每次处理4个fp16
    for (; i + 3 < count; i += 4) {
        float16x4_t vh = vreinterpret_f16_u16(vld1_u16(src + i));
        float32x4_t vf = vcvt_f32_f16(vh);
        vst1q_f32(dst + i, vf);
    }
    
    // 处理剩余元素
    for (; i < count; i++) {
        uint16_t h = src[i];
        uint32_t sign = (h & 0x8000) << 16;
        int32_t exp = ((h >> 10) & 0x1f);
        uint32_t frac = (h & 0x3ff);
        
        if (exp == 0) {
            if (frac == 0) {
                *((uint32_t*)&dst[i]) = sign;  // 0
            } else {
                // 非规格化数
                exp = 1;
                while ((frac & 0x400) == 0) {
                    frac <<= 1;
                    exp--;
                }
                frac &= 0x3ff;
                *((uint32_t*)&dst[i]) = sign | ((exp + 127 - 15) << 23) | (frac << 13);
            }
        } else if (exp == 31) {
            *((uint32_t*)&dst[i]) = sign | 0x7f800000 | (frac << 13);  // inf/nan
        } else {
            *((uint32_t*)&dst[i]) = sign | ((exp + 127 - 15) << 23) | (frac << 13);
        }
    }
}

// ==================== 完全零拷贝的Encoder推理 ====================
int inference_encoder_model_zerocopy(rknn_app_context_t *app_ctx, 
                                     std::vector<float> audio_data, 
                                     float *mel_filters, 
                                     float *encoder_output)
{
    int ret;
    
    if (!app_ctx->use_zero_copy || 
        app_ctx->input_mems == NULL || 
        app_ctx->output_mems == NULL)
    {
       
        // 降级到标准模式
        rknn_input inputs[1];
        rknn_output outputs[1];
        memset(inputs, 0, sizeof(inputs));
        memset(outputs, 0, sizeof(outputs));
        
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_FLOAT32;
        inputs[0].size = N_MELS * ENCODER_INPUT_SIZE * sizeof(float);
        inputs[0].buf = audio_data.data();
        
        ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
        if (ret < 0) {
            printf("rknn_inputs_set fail! ret=%d\n", ret);
            return ret;
        }
        
        ret = rknn_run(app_ctx->rknn_ctx, nullptr);
        if (ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            return ret;
        }
        
        outputs[0].want_float = 1;
        ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
        if (ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            return ret;
        }
        
        memcpy(encoder_output, (float *)outputs[0].buf, ENCODER_OUTPUT_SIZE * sizeof(float));
        rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
        
        return 0;
    }
    
   
    
    // ✅ 完全零拷贝: 将float数据转换为FP16并写入DMA内存
    size_t input_count = N_MELS * ENCODER_INPUT_SIZE;
    
    // 检查模型输入类型
    if (app_ctx->input_attrs[0].type == RKNN_TENSOR_FLOAT16) {
     
        float_to_fp16_neon(
            audio_data.data(),
            (uint16_t*)app_ctx->input_mems[0]->virt_addr,
            input_count
        );
    } else {
        // 如果是FLOAT32,直接拷贝
        memcpy(app_ctx->input_mems[0]->virt_addr, audio_data.data(), 
               input_count * sizeof(float));
    }
    
    // 设置输入(零拷贝)
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->input_mems[0], &app_ctx->input_attrs[0]);
    if (ret < 0) {
        
        return ret;
    }
    
    // 设置输出(零拷贝)
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->output_mems[0], &app_ctx->output_attrs[0]);
    if (ret < 0) {
       
        return ret;
    }
    
    // 运行推理
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return ret;
    }
    
    // ✅ 完全零拷贝: 从DMA内存读取FP16结果并转换为float
    size_t output_count = ENCODER_OUTPUT_SIZE;
    
    if (app_ctx->output_attrs[0].type == RKNN_TENSOR_FLOAT16) {
        
        fp16_to_float_neon(
            (uint16_t*)app_ctx->output_mems[0]->virt_addr,
            encoder_output,
            output_count
        );
    } else {
        // 如果是FLOAT32,直接拷贝
        memcpy(encoder_output, app_ctx->output_mems[0]->virt_addr,
               output_count * sizeof(float));
    }
    
    
    
    return 0;
}

// ==================== 完全零拷贝的Decoder推理 ====================
int inference_decoder_model_full_zerocopy(rknn_app_context_t *app_ctx, 
                                          float *encoder_output, 
                                          VocabEntry *vocab, 
                                          int task_code, 
                                          std::vector<std::string> &recognized_text,
                                          whisper_perf_stats_t *perf_stats)
{
    int ret;
    auto npu_start = std::chrono::high_resolution_clock::now();
    auto npu_end = npu_start;
    double npu_ms = 0.0;

    auto decoder_start = std::chrono::high_resolution_clock::now();
    auto decoder_end = decoder_start;
    double decoder_ms = 0.0;
    int valid_token_count = 0;
    int timestamp_token_count = 0;

    if (!app_ctx->use_zero_copy || 
        app_ctx->input_mems == NULL || 
        app_ctx->output_mems == NULL)
    {
      
        // 降级到标准API...
        return -1;  // 你需要实现标准模式的降级处理
    }

   

    // ✅ 零拷贝: 准备token输入 (直接在DMA内存中操作)
    int64_t *tokens_dma = (int64_t *)app_ctx->input_mems[0]->virt_addr;
    
    // ✅ 零拷贝: 准备encoder输出 (转换为FP16后写入DMA)
    size_t encoder_size = DECODER_INPUT_SIZE;
    if (app_ctx->input_attrs[1].type == RKNN_TENSOR_FLOAT16) {
      
        float_to_fp16_neon(
            encoder_output,
            (uint16_t*)app_ctx->input_mems[1]->virt_addr,
            encoder_size
        );
    } else {
        memcpy(app_ctx->input_mems[1]->virt_addr, encoder_output, 
               encoder_size * sizeof(float));
    }

    // 初始化token序列
    int64_t tokens[MAX_TOKENS + 1] = {50258, task_code, 50359, 50363};
    int timestamp_begin = 50364;
    int next_token = 50258;
    int end_token = 50257;
    int pop_id = MAX_TOKENS;

    for (int i = 0; i < MAX_TOKENS / 4; i++)
    {
        memcpy(&tokens[i * 4], tokens, 4 * sizeof(int64_t));
    }

    int count = 0;
    std::string all_token_str = "";
    bool is_first_token = true;
    int eot_count = 0;
    


    // ✅ 零拷贝: 一次性设置encoder输出(只设置一次,循环外!)
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->input_mems[1], 
                          &app_ctx->input_attrs[1]);
    if (ret < 0) {
       
        return ret;
    }
    
    // ✅ 零拷贝: 一次性设置输出内存(只设置一次,循环外!)
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->output_mems[0], 
                          &app_ctx->output_attrs[0]);
    if (ret < 0) {
       
        return ret;
    }

    // 准备用于FP16→float转换的buffer(如果输出是FP16)
    float *output_buffer = NULL;
    size_t output_size = 12 * 51865;  // tokens * vocab_size
    bool need_fp16_conversion = (app_ctx->output_attrs[0].type == RKNN_TENSOR_FLOAT16);
    
    if (need_fp16_conversion) {
        output_buffer = (float*)malloc(output_size * sizeof(float));
     
    }

    while (next_token != end_token && count < 40)
    {
        count++;

        // ✅ 零拷贝: 直接写入DMA内存
        memcpy(tokens_dma, tokens, MAX_TOKENS * sizeof(int64_t));
        
        // ✅ 零拷贝: 设置token输入(每次循环只需设置tokens)
        ret = rknn_set_io_mem(app_ctx->rknn_ctx, app_ctx->input_mems[0], 
                              &app_ctx->input_attrs[0]);
        if (ret < 0) {
           
            goto out;
        }

        // 运行推理
        npu_start = std::chrono::high_resolution_clock::now();
        ret = rknn_run(app_ctx->rknn_ctx, nullptr);
        npu_end = std::chrono::high_resolution_clock::now();
        npu_ms = std::chrono::duration<double, std::milli>(npu_end - npu_start).count();

        if (ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            goto out;
        }

        // ✅ 零拷贝: 直接从DMA内存读取结果
        float *output_data;
        if (need_fp16_conversion) {
            // FP16→float转换
            fp16_to_float_neon(
                (uint16_t*)app_ctx->output_mems[0]->virt_addr,
                output_buffer,
                output_size
            );
            output_data = output_buffer;
        } else {
            output_data = (float *)app_ctx->output_mems[0]->virt_addr;
        }

        // 找到概率最大的token
        next_token = argmax(output_data);

        

        std::string next_token_str = vocab[next_token].token;
        all_token_str += next_token_str;

        // 统计token类型
        if (next_token > timestamp_begin)
        {
            timestamp_token_count++;
            if (count <= 10) {
            
            }
            continue;
        }
        
        if (next_token == end_token) {
            eot_count++;
       
        }
        
        if (next_token != end_token)
        {
            valid_token_count++;
          
        }

        if (pop_id > 4)
        {
            pop_id--;
        }

        tokens[MAX_TOKENS] = next_token;
        for (int j = pop_id; j < MAX_TOKENS; j++)
        {
            tokens[j] = tokens[j + 1];
        }
    }



    decoder_end = std::chrono::high_resolution_clock::now();
    decoder_ms = std::chrono::duration<double, std::milli>(decoder_end - decoder_start).count();
    
    if (perf_stats != NULL)
    {
        perf_stats->total_loops = count;
        perf_stats->valid_tokens = valid_token_count;
        perf_stats->timestamp_tokens = timestamp_token_count;
        perf_stats->decoder_time_ms = decoder_ms;
    }

    replace_substr(all_token_str, "\u0120", " ");
    replace_substr(all_token_str, "<|endoftext|>", "");
    replace_substr(all_token_str, "\n", "");

    if (all_token_str.size())
    {
        try {
            all_token_str = base64_decode(all_token_str);
        } 
        catch(...) {
            // 忽略异常
        }
        recognized_text.push_back(all_token_str);
    } else {
      
    }

out:
    if (output_buffer != NULL) {
        free(output_buffer);
    }


    return ret;
}

// ==================== 零拷贝优化版本 ====================
// 结构体定义已在 whisper.h 中更新

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    char dims_str[100];
    char temp_str[100];
    memset(dims_str, 0, sizeof(dims_str));
    for (int i = 0; i < attr->n_dims; i++)
    {
        strcpy(temp_str, dims_str);
        if (i == attr->n_dims - 1)
        {
            sprintf(dims_str, "%s%d", temp_str, attr->dims[i]);
        }
        else
        {
            sprintf(dims_str, "%s%d, ", temp_str, attr->dims[i]);
        }
    }

    printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, dims_str, attr->n_elems, attr->size, get_format_string(attr->fmt),
           get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

// ==================== 零拷贝：初始化模型 ====================
int init_whisper_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    rknn_context ctx = 0;

    // Load RKNN Model
    ret = rknn_init(&ctx, (void *)model_path, model_len, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    // ✅ 零拷贝：分配DMA内存
  
    
    // 分配输入内存
    app_ctx->input_mems = (rknn_tensor_mem **)malloc(io_num.n_input * sizeof(rknn_tensor_mem *));
    for (int i = 0; i < io_num.n_input; i++)
    {
        app_ctx->input_mems[i] = rknn_create_mem(ctx, input_attrs[i].size_with_stride);
        if (app_ctx->input_mems[i] == NULL)
        {
          
            app_ctx->use_zero_copy = false;
            return -1;
        }
       
    }
    
    // 分配输出内存
    app_ctx->output_mems = (rknn_tensor_mem **)malloc(io_num.n_output * sizeof(rknn_tensor_mem *));
    for (int i = 0; i < io_num.n_output; i++)
    {
        app_ctx->output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
        if (app_ctx->output_mems[i] == NULL)
        {
         
            app_ctx->use_zero_copy = false;
            return -1;
        }
       
              
    }
    
    app_ctx->use_zero_copy = true;
 

    return 0;
}

// ==================== 零拷贝：释放模型 ====================
int release_whisper_model(rknn_app_context_t *app_ctx)
{
    // 释放零拷贝内存
    if (app_ctx->use_zero_copy)
    {
  
        
        if (app_ctx->input_mems != NULL)
        {
            for (int i = 0; i < app_ctx->io_num.n_input; i++)
            {
                if (app_ctx->input_mems[i] != NULL)
                {
                    rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mems[i]);
                }
            }
            free(app_ctx->input_mems);
            app_ctx->input_mems = NULL;
        }
        
        if (app_ctx->output_mems != NULL)
        {
            for (int i = 0; i < app_ctx->io_num.n_output; i++)
            {
                if (app_ctx->output_mems[i] != NULL)
                {
                    rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->output_mems[i]);
                }
            }
            free(app_ctx->output_mems);
            app_ctx->output_mems = NULL;
        }
    }
    
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// ==================== 零拷贝：Encoder推理 ====================
int inference_encoder_model(rknn_app_context_t *app_ctx, std::vector<float> audio_data, 
                           float *mel_filters, float *encoder_output)
{
       // 尝试使用零拷贝模式
       return inference_encoder_model_zerocopy(app_ctx, audio_data, 
                                               mel_filters, encoder_output);
   }

// ==================== 零拷贝：Decoder推理 ====================
int inference_decoder_model(rknn_app_context_t *app_ctx, float *encoder_output, 
                           VocabEntry *vocab, int task_code, 
                           std::vector<std::string> &recognized_text,
                           whisper_perf_stats_t *perf_stats)
{

    // 尝试使用完全零拷贝模式
    int ret = inference_decoder_model_full_zerocopy(
        app_ctx, encoder_output, vocab, task_code, 
        recognized_text, perf_stats
    );

    

    //int ret;
    rknn_input inputs[2];
    rknn_output outputs[1];

    auto npu_start = std::chrono::high_resolution_clock::now();
    auto npu_end = std::chrono::high_resolution_clock::now();
    double npu_ms = std::chrono::duration<double, std::milli>(npu_end - npu_start).count();

    auto decoder_start = std::chrono::high_resolution_clock::now();
    auto decoder_end = decoder_start;
    double decoder_ms = 0.0;
    int valid_token_count = 0;
    int timestamp_token_count = 0;

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_INT64;
    inputs[0].size = MAX_TOKENS * sizeof(int64_t);
    inputs[0].buf = (int64_t *)malloc(inputs[0].size);

    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;  // 使用FLOAT32,驱动自动转FP16
    inputs[1].size = DECODER_INPUT_SIZE * sizeof(float);
    inputs[1].buf = (float *)malloc(inputs[1].size);
    memcpy(inputs[1].buf, encoder_output, inputs[1].size);

    int64_t tokens[MAX_TOKENS + 1] = {50258, task_code, 50359, 50363};
    int timestamp_begin = 50364;
    int next_token = 50258;
    int end_token = 50257;
    int pop_id = MAX_TOKENS;

    int count = 0;
    std::string all_token_str = "";

    // 初始化token序列
    for (int i = 0; i < MAX_TOKENS / 4; i++)
    {
        memcpy(&tokens[i * 4], tokens, 4 * sizeof(int64_t));
    }

    bool is_first_token = true;
    int eot_count = 0;
   

   

    while (next_token != end_token && count < 40)
    {
        count++;

        memcpy(inputs[0].buf, tokens, inputs[0].size);

        ret = rknn_inputs_set(app_ctx->rknn_ctx, 2, inputs);
        if (ret < 0)
        {
            printf("rknn_input_set fail! ret=%d\n", ret);
            goto out;
        }

        // Run
        npu_start = std::chrono::high_resolution_clock::now();
        ret = rknn_run(app_ctx->rknn_ctx, nullptr);
        npu_end = std::chrono::high_resolution_clock::now();
        npu_ms = std::chrono::duration<double, std::milli>(npu_end - npu_start).count();

        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d\n", ret);
            goto out;
        }

        // Get Output
        outputs[0].want_float = 1;
        ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
        if (ret < 0)
        {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            goto out;
        }

        next_token = argmax((float *)outputs[0].buf);



        std::string next_token_str = vocab[next_token].token;
        all_token_str += next_token_str;

        // 统计token类型
        if (next_token > timestamp_begin)
        {
            timestamp_token_count++;
            
            rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
            continue;
        }
        
        if (next_token == end_token) {
            eot_count++;
           
        }
        
        if (next_token != end_token)
        {
            valid_token_count++;
           
        }

        if (pop_id > 4)
        {
            pop_id--;
        }

        tokens[MAX_TOKENS] = next_token;

        for (int j = pop_id; j < MAX_TOKENS; j++)
        {
            tokens[j] = tokens[j + 1];
        }

        rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);
    }

  

    decoder_end = std::chrono::high_resolution_clock::now();
    decoder_ms = std::chrono::duration<double, std::milli>(decoder_end - decoder_start).count();
    
    if (perf_stats != NULL)
    {
        perf_stats->total_loops = count;
        perf_stats->valid_tokens = valid_token_count;
        perf_stats->timestamp_tokens = timestamp_token_count;
        perf_stats->decoder_time_ms = decoder_ms;
    }

    replace_substr(all_token_str, "\u0120", " ");
    replace_substr(all_token_str, "<|endoftext|>", "");
    replace_substr(all_token_str, "\n", "");

    if (all_token_str.size())
    {
        try 
        {
            all_token_str = base64_decode(all_token_str);
        } 
        catch(...) {
            // 忽略异常
        }
        recognized_text.push_back(all_token_str);
    } else {
       
    }

out:
    for (int i = 0; i < 2; i++)
    {
        if (inputs[i].buf != NULL)
        {
            free(inputs[i].buf);
        }
    }

    return ret;
}

// ==================== 完整推理流程 ====================
int inference_whisper_model(rknn_whisper_context_t *app_ctx, 
                           std::vector<float> audio_data, 
                           float *mel_filters, 
                           VocabEntry *vocab, 
                           int task_code, 
                           std::vector<std::string> &recognized_text,
                           whisper_perf_stats_t *perf_stats)
{
    int ret;
    float *encoder_output = (float *)malloc(ENCODER_OUTPUT_SIZE * sizeof(float));

    auto total_start = std::chrono::high_resolution_clock::now();
    auto encoder_start = total_start;
    auto encoder_end = total_start;
    auto total_end = total_start;
    double encoder_ms = 0.0;
    double total_ms = 0.0;

    recognized_text.clear();

    // ===== Encoder =====
    encoder_start = std::chrono::high_resolution_clock::now();
    ret = inference_encoder_model(&app_ctx->encoder_context, audio_data, mel_filters, encoder_output);
    encoder_end = std::chrono::high_resolution_clock::now();
    
    if (ret != 0)
    {
        printf("inference_encoder_model fail! ret=%d\n", ret);
        goto out;
    }
    
    encoder_ms = std::chrono::duration<double, std::milli>(encoder_end - encoder_start).count();

    // ===== Decoder =====
    ret = inference_decoder_model(&app_ctx->decoder_context, encoder_output, vocab, task_code, 
                                  recognized_text, perf_stats);
    if (ret != 0)
    {
        printf("inference_decoder_model fail! ret=%d\n", ret);
        goto out;
    }

    total_end = std::chrono::high_resolution_clock::now();
    total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();


out:
    if (encoder_output != NULL)
    {
        free(encoder_output);
    }

    return ret;
}