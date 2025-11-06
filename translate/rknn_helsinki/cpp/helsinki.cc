#include "helsinki.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <arm_neon.h>

// ==================== FP16转换函数 (NEON优化) ====================

void HelsinkiTranslator::float_to_fp16_neon(const float* src, uint16_t* dst, size_t count) {
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
        uint32_t f = *((uint32_t*)&src[i]);
        uint32_t sign = (f >> 16) & 0x8000;
        int32_t exp = ((f >> 23) & 0xff) - 127 + 15;
        uint32_t frac = (f >> 13) & 0x3ff;
        
        if (exp <= 0) {
            dst[i] = sign;
        } else if (exp >= 31) {
            dst[i] = sign | 0x7c00;
        } else {
            dst[i] = sign | (exp << 10) | frac;
        }
    }
}

void HelsinkiTranslator::fp16_to_float_neon(const uint16_t* src, float* dst, size_t count) {
    size_t i = 0;
    
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    for (; i + 7 < count; i += 8) {
        float16x4_t h1 = vreinterpret_f16_u16(vld1_u16(src + i));
        float16x4_t h2 = vreinterpret_f16_u16(vld1_u16(src + i + 4));
        
        float32x4_t v1 = vcvt_f32_f16(h1);
        float32x4_t v2 = vcvt_f32_f16(h2);
        
        vst1q_f32(dst + i, v1);
        vst1q_f32(dst + i + 4, v2);
    }
#endif
    
    for (; i + 3 < count; i += 4) {
        float16x4_t vh = vreinterpret_f16_u16(vld1_u16(src + i));
        float32x4_t vf = vcvt_f32_f16(vh);
        vst1q_f32(dst + i, vf);
    }
    
    for (; i < count; i++) {
        uint16_t h = src[i];
        uint32_t sign = (h & 0x8000) << 16;
        int32_t exp = ((h >> 10) & 0x1f);
        uint32_t frac = (h & 0x3ff);
        
        if (exp == 0) {
            if (frac == 0) {
                *((uint32_t*)&dst[i]) = sign;
            } else {
                exp = 1;
                while ((frac & 0x400) == 0) {
                    frac <<= 1;
                    exp--;
                }
                frac &= 0x3ff;
                *((uint32_t*)&dst[i]) = sign | ((exp + 127 - 15) << 23) | (frac << 13);
            }
        } else if (exp == 31) {
            *((uint32_t*)&dst[i]) = sign | 0x7f800000 | (frac << 13);
        } else {
            *((uint32_t*)&dst[i]) = sign | ((exp + 127 - 15) << 23) | (frac << 13);
        }
    }
}

// ==================== 构造/析构 ====================

HelsinkiTranslator::HelsinkiTranslator()
    : encoder_ctx_(0),
      decoder_ctx_(0),
      encoder_seq_len_(64),
      decoder_seq_len_(64),
      hidden_size_(512),
      vocab_size_(65001),
      pad_token_id_(65000),
      eos_token_id_(0),
      decoder_start_token_id_(65000),
      initialized_(false),
      use_zero_copy_(false),
      encoder_input_mems_(nullptr),
      encoder_output_mems_(nullptr),
      encoder_input_attrs_(nullptr),
      encoder_output_attrs_(nullptr),
      decoder_input_mems_(nullptr),
      decoder_output_mems_(nullptr),
      decoder_input_attrs_(nullptr),
      decoder_output_attrs_(nullptr),
      encoder_input_num_(0),
      encoder_output_num_(0),
      decoder_input_num_(0),
      decoder_output_num_(0) {}

HelsinkiTranslator::~HelsinkiTranslator() {
    release();
}

// ==================== Zero-Copy 内存初始化 ====================

int HelsinkiTranslator::init_zero_copy_memory() {
    std::cout << "[INFO] 初始化 Zero-Copy 内存...\n";
    
    // ===== Encoder =====
    rknn_input_output_num encoder_io_num;
    int ret = rknn_query(encoder_ctx_, RKNN_QUERY_IN_OUT_NUM, &encoder_io_num, sizeof(encoder_io_num));
    if (ret != 0) {
        std::cerr << "[WARN] 查询 Encoder IO 数量失败，降级到标准模式\n";
        return -1;
    }
    
    encoder_input_num_ = encoder_io_num.n_input;
    encoder_output_num_ = encoder_io_num.n_output;
    
    // 分配 Encoder 输入属性
    encoder_input_attrs_ = new rknn_tensor_attr[encoder_input_num_];
    for (int i = 0; i < encoder_input_num_; i++) {
        encoder_input_attrs_[i].index = i;
        ret = rknn_query(encoder_ctx_, RKNN_QUERY_INPUT_ATTR, &encoder_input_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != 0) {
            std::cerr << "[WARN] 查询 Encoder 输入属性失败\n";
            return -1;
        }
    }
    
    // 分配 Encoder 输出属性
    encoder_output_attrs_ = new rknn_tensor_attr[encoder_output_num_];
    for (int i = 0; i < encoder_output_num_; i++) {
        encoder_output_attrs_[i].index = i;
        ret = rknn_query(encoder_ctx_, RKNN_QUERY_OUTPUT_ATTR, &encoder_output_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != 0) {
            std::cerr << "[WARN] 查询 Encoder 输出属性失败\n";
            return -1;
        }
    }
    
    // 分配 Encoder DMA 内存
    encoder_input_mems_ = new rknn_tensor_mem*[encoder_input_num_];
    for (int i = 0; i < encoder_input_num_; i++) {
        encoder_input_mems_[i] = rknn_create_mem(encoder_ctx_, encoder_input_attrs_[i].size_with_stride);
        if (encoder_input_mems_[i] == nullptr) {
            std::cerr << "[WARN] 创建 Encoder 输入 DMA 内存失败\n";
            return -1;
        }
        std::cout << "[INFO] Encoder Input[" << i << "] DMA: " 
                  << encoder_input_attrs_[i].size_with_stride << " bytes\n";
    }
    
    encoder_output_mems_ = new rknn_tensor_mem*[encoder_output_num_];
    for (int i = 0; i < encoder_output_num_; i++) {
        encoder_output_mems_[i] = rknn_create_mem(encoder_ctx_, encoder_output_attrs_[i].size_with_stride);
        if (encoder_output_mems_[i] == nullptr) {
            std::cerr << "[WARN] 创建 Encoder 输出 DMA 内存失败\n";
            return -1;
        }
        std::cout << "[INFO] Encoder Output[" << i << "] DMA: " 
                  << encoder_output_attrs_[i].size_with_stride << " bytes\n";
    }
    
    // ===== Decoder =====
    rknn_input_output_num decoder_io_num;
    ret = rknn_query(decoder_ctx_, RKNN_QUERY_IN_OUT_NUM, &decoder_io_num, sizeof(decoder_io_num));
    if (ret != 0) {
        std::cerr << "[WARN] 查询 Decoder IO 数量失败\n";
        return -1;
    }
    
    decoder_input_num_ = decoder_io_num.n_input;
    decoder_output_num_ = decoder_io_num.n_output;
    
    // 分配 Decoder 输入属性
    decoder_input_attrs_ = new rknn_tensor_attr[decoder_input_num_];
    for (int i = 0; i < decoder_input_num_; i++) {
        decoder_input_attrs_[i].index = i;
        ret = rknn_query(decoder_ctx_, RKNN_QUERY_INPUT_ATTR, &decoder_input_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != 0) {
            std::cerr << "[WARN] 查询 Decoder 输入属性失败\n";
            return -1;
        }
    }
    
    // 分配 Decoder 输出属性
    decoder_output_attrs_ = new rknn_tensor_attr[decoder_output_num_];
    for (int i = 0; i < decoder_output_num_; i++) {
        decoder_output_attrs_[i].index = i;
        ret = rknn_query(decoder_ctx_, RKNN_QUERY_OUTPUT_ATTR, &decoder_output_attrs_[i], sizeof(rknn_tensor_attr));
        if (ret != 0) {
            std::cerr << "[WARN] 查询 Decoder 输出属性失败\n";
            return -1;
        }
    }
    
    // 分配 Decoder DMA 内存
    decoder_input_mems_ = new rknn_tensor_mem*[decoder_input_num_];
    for (int i = 0; i < decoder_input_num_; i++) {
        decoder_input_mems_[i] = rknn_create_mem(decoder_ctx_, decoder_input_attrs_[i].size_with_stride);
        if (decoder_input_mems_[i] == nullptr) {
            std::cerr << "[WARN] 创建 Decoder 输入 DMA 内存失败\n";
            return -1;
        }
        std::cout << "[INFO] Decoder Input[" << i << "] DMA: " 
                  << decoder_input_attrs_[i].size_with_stride << " bytes\n";
    }
    
    decoder_output_mems_ = new rknn_tensor_mem*[decoder_output_num_];
    for (int i = 0; i < decoder_output_num_; i++) {
        decoder_output_mems_[i] = rknn_create_mem(decoder_ctx_, decoder_output_attrs_[i].size_with_stride);
            if (decoder_output_mems_[i] == nullptr) {
            std::cerr << "[WARN] 创建 Decoder 输出 DMA 内存失败\n";
            return -1;
        }
        std::cout << "[INFO] Decoder Output[" << i << "] DMA: " 
                  << decoder_output_attrs_[i].size_with_stride << " bytes\n";
    }
    
    std::cout << "[INFO] ✅ Zero-Copy 内存初始化成功\n";
    use_zero_copy_ = true;
    return 0;
}

// ==================== Zero-Copy 内存释放 ====================

void HelsinkiTranslator::release_zero_copy_memory() {
    if (!use_zero_copy_) return;
    
    std::cout << "[INFO] 释放 Zero-Copy 内存...\n";
    
    // 释放 Encoder
    if (encoder_input_mems_) {
        for (int i = 0; i < encoder_input_num_; i++) {
            if (encoder_input_mems_[i]) {
                rknn_destroy_mem(encoder_ctx_, encoder_input_mems_[i]);
            }
        }
        delete[] encoder_input_mems_;
        encoder_input_mems_ = nullptr;
    }
    
    if (encoder_output_mems_) {
        for (int i = 0; i < encoder_output_num_; i++) {
            if (encoder_output_mems_[i]) {
                rknn_destroy_mem(encoder_ctx_, encoder_output_mems_[i]);
            }
        }
        delete[] encoder_output_mems_;
        encoder_output_mems_ = nullptr;
    }
    
    // 释放 Decoder
    if (decoder_input_mems_) {
        for (int i = 0; i < decoder_input_num_; i++) {
            if (decoder_input_mems_[i]) {
                rknn_destroy_mem(decoder_ctx_, decoder_input_mems_[i]);
            }
        }
        delete[] decoder_input_mems_;
        decoder_input_mems_ = nullptr;
    }
    
    if (decoder_output_mems_) {
        for (int i = 0; i < decoder_output_num_; i++) {
            if (decoder_output_mems_[i]) {
                rknn_destroy_mem(decoder_ctx_, decoder_output_mems_[i]);
            }
        }
        delete[] decoder_output_mems_;
        decoder_output_mems_ = nullptr;
    }
    
    // 释放属性
    if (encoder_input_attrs_) {
        delete[] encoder_input_attrs_;
        encoder_input_attrs_ = nullptr;
    }
    if (encoder_output_attrs_) {
        delete[] encoder_output_attrs_;
        encoder_output_attrs_ = nullptr;
    }
    if (decoder_input_attrs_) {
        delete[] decoder_input_attrs_;
        decoder_input_attrs_ = nullptr;
    }
    if (decoder_output_attrs_) {
        delete[] decoder_output_attrs_;
        decoder_output_attrs_ = nullptr;
    }
    
    use_zero_copy_ = false;
}

// ==================== 初始化模型 ====================

int HelsinkiTranslator::init(const char* encoder_path, const char* decoder_path) {
    if (initialized_) {
        std::cerr << "[WARN] 模型已初始化，重复初始化被忽略\n";
        return 0;
    }

    int ret;

    // ===== 加载 Encoder 模型 =====
    FILE* fp_encoder = fopen(encoder_path, "rb");
    if (fp_encoder == NULL) {
        std::cerr << "[ERROR] 无法打开 Encoder 模型文件: " << encoder_path << "\n";
        return -1;
    }
    fseek(fp_encoder, 0, SEEK_END);
    long encoder_size = ftell(fp_encoder);
    fseek(fp_encoder, 0, SEEK_SET);
    
    void* encoder_data = malloc(encoder_size);
    if (encoder_data == NULL) {
        std::cerr << "[ERROR] Encoder 内存分配失败\n";
        fclose(fp_encoder);
        return -1;
    }
    
    size_t read_size = fread(encoder_data, 1, encoder_size, fp_encoder);
    fclose(fp_encoder);
    
    if (read_size != (size_t)encoder_size) {
        std::cerr << "[ERROR] 读取 Encoder 模型文件失败\n";
        free(encoder_data);
        return -1;
    }

    std::cout << "[INFO] Encoder 模型文件大小: " << encoder_size << " bytes\n";

    ret = rknn_init(&encoder_ctx_, encoder_data, encoder_size, 0, nullptr);
    free(encoder_data);
    
    if (ret < 0) {
        std::cerr << "[ERROR] 加载 Encoder 模型失败，错误码: " << ret << "\n";
        return -1;
    }

    std::cout << "[INFO] Encoder 模型加载成功\n";

    // ===== 加载 Decoder 模型 =====
    FILE* fp_decoder = fopen(decoder_path, "rb");
    if (fp_decoder == NULL) {
        std::cerr << "[ERROR] 无法打开 Decoder 模型文件: " << decoder_path << "\n";
        rknn_destroy(encoder_ctx_);
        encoder_ctx_ = 0;
        return -1;
    }
    fseek(fp_decoder, 0, SEEK_END);
    long decoder_size = ftell(fp_decoder);
    fseek(fp_decoder, 0, SEEK_SET);
    
    void* decoder_data = malloc(decoder_size);
    if (decoder_data == NULL) {
        std::cerr << "[ERROR] Decoder 内存分配失败\n";
        fclose(fp_decoder);
        rknn_destroy(encoder_ctx_);
        encoder_ctx_ = 0;
        return -1;
    }
    
    read_size = fread(decoder_data, 1, decoder_size, fp_decoder);
    fclose(fp_decoder);
    
    if (read_size != (size_t)decoder_size) {
        std::cerr << "[ERROR] 读取 Decoder 模型文件失败\n";
        free(decoder_data);
        rknn_destroy(encoder_ctx_);
        encoder_ctx_ = 0;
        return -1;
    }

    std::cout << "[INFO] Decoder 模型文件大小: " << decoder_size << " bytes\n";

    ret = rknn_init(&decoder_ctx_, decoder_data, decoder_size, 0, nullptr);
    free(decoder_data);
    
    if (ret < 0) {
        std::cerr << "[ERROR] 加载 Decoder 模型失败，错误码: " << ret << "\n";
        rknn_destroy(encoder_ctx_);
        encoder_ctx_ = 0;
        return -1;
    }

    std::cout << "[INFO] Decoder 模型加载成功\n";

    initialized_ = true;
    
    // ===== 初始化 Zero-Copy 内存 =====
    ret = init_zero_copy_memory();
    if (ret < 0) {
        std::cerr << "[WARN] Zero-Copy 初始化失败，将使用标准模式\n";
        use_zero_copy_ = false;
    }
    
    // 打印模型信息
    print_model_info();
    
    std::cout << "[INFO] ✅ 模型初始化成功 (Zero-Copy: " 
              << (use_zero_copy_ ? "启用" : "禁用") << ")\n";
    
    return 0;
}

// ==================== Encoder 推理 (Zero-Copy) ====================

int HelsinkiTranslator::run_encoder_zerocopy(const int64_t* input_ids, 
                                             const int64_t* attention_mask,
                                             int input_len, 
                                             float* encoder_output) {
    std::cout << "[DEBUG] [ZeroCopy] Running encoder with input_len: " << input_len << "\n";
    
    // ✅ 1. 直接写入 DMA 内存（INT64 不需要类型转换）
    memcpy(encoder_input_mems_[0]->virt_addr, input_ids, input_len * sizeof(int64_t));
    memcpy(encoder_input_mems_[1]->virt_addr, attention_mask, input_len * sizeof(int64_t));
    
    // ✅ 2. 设置零拷贝输入
    int ret = rknn_set_io_mem(encoder_ctx_, encoder_input_mems_[0], &encoder_input_attrs_[0]);
    if (ret < 0) {
        std::cerr << "[ERROR] 设置 Encoder 输入0失败，错误码: " << ret << "\n";
        return -1;
    }
    
    ret = rknn_set_io_mem(encoder_ctx_, encoder_input_mems_[1], &encoder_input_attrs_[1]);
    if (ret < 0) {
        std::cerr << "[ERROR] 设置 Encoder 输入1失败，错误码: " << ret << "\n";
        return -1;
    }
    
    // ✅ 3. 设置零拷贝输出
    ret = rknn_set_io_mem(encoder_ctx_, encoder_output_mems_[0], &encoder_output_attrs_[0]);
    if (ret < 0) {
        std::cerr << "[ERROR] 设置 Encoder 输出失败，错误码: " << ret << "\n";
        return -1;
    }
    
    // ✅ 4. 运行推理
    ret = rknn_run(encoder_ctx_, nullptr);
    if (ret < 0) {
        std::cerr << "[ERROR] Encoder 推理失败，错误码: " << ret << "\n";
        return -1;
    }
    
    // ✅ 5. 从 DMA 内存读取结果
    size_t output_size = encoder_seq_len_ * hidden_size_;
    
    if (encoder_output_attrs_[0].type == RKNN_TENSOR_FLOAT16) {
        // FP16 → FP32 转换
        std::cout << "[DEBUG] 转换 Encoder 输出: FP16 → FP32\n";
        fp16_to_float_neon((uint16_t*)encoder_output_mems_[0]->virt_addr, 
                          encoder_output, 
                          output_size);
    } else {
        // 直接拷贝 FP32
        memcpy(encoder_output, encoder_output_mems_[0]->virt_addr, 
               output_size * sizeof(float));
    }
    
    // 验证输出
    float* enc_data = encoder_output;
    float sum = 0, min_val = enc_data[0], max_val = enc_data[0];
    for (size_t i = 0; i < output_size; i++) {
        sum += enc_data[i];
        if (enc_data[i] < min_val) min_val = enc_data[i];
        if (enc_data[i] > max_val) max_val = enc_data[i];
    }
    
    std::cout << "\n[CHECKPOINT] Encoder Output (ZeroCopy):\n";
    std::cout << "  Mean: " << (sum/output_size) << "\n";
    std::cout << "  Min:  " << min_val << "\n";
    std::cout << "  Max:  " << max_val << "\n";
    std::cout << "  First 5: [" << enc_data[0] << ", " << enc_data[1] 
              << ", " << enc_data[2] << ", " << enc_data[3] 
              << ", " << enc_data[4] << "]\n\n";
    
    std::cout << "[DEBUG] ✅ Encoder 执行成功 (Zero-Copy)\n";
    return 0;
}

// ==================== Encoder 推理 (标准模式，降级) ====================

int HelsinkiTranslator::run_encoder(const int64_t* input_ids, 
                                    const int64_t* attention_mask,
                                    int input_len, 
                                    float* encoder_output) {
    // 如果启用了 Zero-Copy，优先使用
    if (use_zero_copy_) {
        return run_encoder_zerocopy(input_ids, attention_mask, input_len, encoder_output);
    }
    
    std::cout << "[DEBUG] Running encoder (standard mode) with input_len: " << input_len << "\n";
    
    rknn_input inputs[2];
    memset(inputs, 0, sizeof(inputs));
    
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_INT64;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = (void*)input_ids;
    inputs[0].size = input_len * sizeof(int64_t);
    inputs[0].pass_through = 0;
    
    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_INT64;
    inputs[1].fmt = RKNN_TENSOR_NHWC;
    inputs[1].buf = (void*)attention_mask;
    inputs[1].size = input_len * sizeof(int64_t);
    inputs[1].pass_through = 0;

    int ret = rknn_inputs_set(encoder_ctx_, 2, inputs);
    if (ret < 0) {
        std::cerr << "[ERROR] 设置 Encoder 输入失败，错误码: " << ret << "\n";
        return -1;
    }

    ret = rknn_run(encoder_ctx_, nullptr);
    if (ret < 0) {
        std::cerr << "[ERROR] Encoder 推理失败，错误码: " << ret << "\n";
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;

    ret = rknn_outputs_get(encoder_ctx_, 1, outputs, NULL);
    if (ret < 0) {
        std::cerr << "[ERROR] 获取 Encoder 输出失败，错误码: " << ret << "\n";
        return -1;
    }

    size_t expected_size = encoder_seq_len_ * hidden_size_ * sizeof(float);
    size_t copy_size = (outputs[0].size < expected_size) ? outputs[0].size : expected_size;
    memcpy(encoder_output, outputs[0].buf, copy_size);
    
    rknn_outputs_release(encoder_ctx_, 1, outputs);
    
    std::cout << "[DEBUG] Encoder 执行成功 (standard mode)\n";
    return 0;
}

// ==================== Decoder 推理 (Zero-Copy) ====================

int HelsinkiTranslator::run_decoder_zerocopy(const int64_t* decoder_input_ids, 
                                             int decoder_input_len,
                                             const float* encoder_hidden_states,
                                             const int64_t* encoder_attention_mask,
                                             float* logits) {
    const int DECODER_FIXED_LEN = 64;
    
    // ✅ 1. 准备 decoder_input_ids（padding 到 64）
    std::vector<int64_t> padded_decoder_input_ids(DECODER_FIXED_LEN, pad_token_id_);
    int copy_len = std::min(decoder_input_len, DECODER_FIXED_LEN);
    for (int i = 0; i < copy_len; i++) {
        padded_decoder_input_ids[i] = decoder_input_ids[i];
    }
    
    // ✅ 2. 写入 DMA 内存 (Input 0: decoder_input_ids)
    memcpy(decoder_input_mems_[0]->virt_addr, padded_decoder_input_ids.data(), 
           DECODER_FIXED_LEN * sizeof(int64_t));
    
    // ✅ 3. 设置输入 (只需设置 Input 0，Input 1和2在外部已设置)
    int ret = rknn_set_io_mem(decoder_ctx_, decoder_input_mems_[0], &decoder_input_attrs_[0]);
    if (ret < 0) {
        std::cerr << "[ERROR] 设置 Decoder 输入0失败，错误码: " << ret << "\n";
        return -1;
    }
    
    // ✅ 4. 运行推理
    ret = rknn_run(decoder_ctx_, nullptr);
    if (ret < 0) {
        std::cerr << "[ERROR] Decoder 推理失败，错误码: " << ret << "\n";
        return -1;
    }
    
    // ✅ 5. 从 DMA 内存读取结果
    int extract_pos = std::min(decoder_input_len - 1, DECODER_FIXED_LEN - 1);
    if (extract_pos < 0) extract_pos = 0;
    
    std::cout << "[DEBUG] [ZeroCopy] decoder_input_len=" << decoder_input_len 
              << ", extracting from position " << extract_pos << "\n";
    
    size_t offset = extract_pos * vocab_size_;
    
    if (decoder_output_attrs_[0].type == RKNN_TENSOR_FLOAT16) {
        // FP16 → FP32 转换
        std::cout << "[DEBUG] 转换 Decoder 输出: FP16 → FP32\n";
        uint16_t* fp16_output = (uint16_t*)decoder_output_mems_[0]->virt_addr;
        std::vector<float> temp_output(DECODER_FIXED_LEN * vocab_size_);
        fp16_to_float_neon(fp16_output, temp_output.data(), DECODER_FIXED_LEN * vocab_size_);
        memcpy(logits, temp_output.data() + offset, vocab_size_ * sizeof(float));
    } else {
        // 直接拷贝 FP32
        float* output_ptr = (float*)decoder_output_mems_[0]->virt_addr;
        memcpy(logits, output_ptr + offset, vocab_size_ * sizeof(float));
    }
    
    // 验证提取的 logits
    std::cout << "[DEBUG] Extracted logits verification (ZeroCopy):\n";
    std::cout << "  logits[0]=" << logits[0] << "\n";
    std::cout << "  logits[8]=" << logits[8] << "\n";
    std::cout << "  logits[2]=" << logits[2] << "\n";
    
    return 0;
}

// ==================== Decoder 推理 (标准模式，降级) ====================

int HelsinkiTranslator::run_decoder(const int64_t* decoder_input_ids, 
                                    int decoder_input_len,
                                    const float* encoder_hidden_states,
                                    const int64_t* encoder_attention_mask,
                                    float* logits) {
    // 如果启用了 Zero-Copy，优先使用
    if (use_zero_copy_) {
        return run_decoder_zerocopy(decoder_input_ids, decoder_input_len, 
                                   encoder_hidden_states, encoder_attention_mask, logits);
    }
    
    const int DECODER_FIXED_LEN = 64;
    
    std::vector<int64_t> padded_decoder_input_ids(DECODER_FIXED_LEN, pad_token_id_);
    int copy_len = std::min(decoder_input_len, DECODER_FIXED_LEN);
    for (int i = 0; i < copy_len; i++) {
        padded_decoder_input_ids[i] = decoder_input_ids[i];
    }
    
    rknn_input inputs[3];
    memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_INT64;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = (void*)padded_decoder_input_ids.data();
    inputs[0].size = DECODER_FIXED_LEN * sizeof(int64_t);
    inputs[0].pass_through = 0;

    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].fmt = RKNN_TENSOR_NHWC;
    inputs[1].buf = (void*)encoder_hidden_states;
    inputs[1].size = encoder_seq_len_ * hidden_size_ * sizeof(float);
    inputs[1].pass_through = 0;
    
    inputs[2].index = 2;
    inputs[2].type = RKNN_TENSOR_INT64;
    inputs[2].fmt = RKNN_TENSOR_NHWC;
    inputs[2].buf = (void*)encoder_attention_mask;
    inputs[2].size = encoder_seq_len_ * sizeof(int64_t);
    inputs[2].pass_through = 0;

    int ret = rknn_inputs_set(decoder_ctx_, 3, inputs);
    if (ret < 0) {
        std::cerr << "[ERROR] 设置 Decoder 输入失败，错误码: " << ret << "\n";
        return -1;
    }

    ret = rknn_run(decoder_ctx_, nullptr);
    if (ret < 0) {
        std::cerr << "[ERROR] Decoder 推理失败，错误码: " << ret << "\n";
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;

    ret = rknn_outputs_get(decoder_ctx_, 1, outputs, NULL);
    if (ret < 0) {
        std::cerr << "[ERROR] 获取 Decoder 输出失败，错误码: " << ret << "\n";
        return -1;
    }

    int extract_pos = std::min(decoder_input_len - 1, DECODER_FIXED_LEN - 1);
    if (extract_pos < 0) extract_pos = 0;
    
    float* output_ptr = (float*)outputs[0].buf;
    size_t offset = extract_pos * vocab_size_;
    memcpy(logits, output_ptr + offset, vocab_size_ * sizeof(float));
    
    rknn_outputs_release(decoder_ctx_, 1, outputs);
    return 0;
}

// ==================== 翻译主函数 (Zero-Copy 优化版) ====================

int HelsinkiTranslator::translate(const int64_t* input_ids, 
                                  const int64_t* attention_mask,
                                  int seq_len, 
                                  int64_t* output_ids, 
                                  int max_output_len,
                                  helsinki_perf_stats_t* perf_stats) {
    if (!initialized_) {
        std::cerr << "[ERROR] 模型未初始化\n";
        return -1;
    }

    std::cout << "[INFO] 开始翻译 (Zero-Copy: " 
              << (use_zero_copy_ ? "启用" : "禁用") << ")\n";

    auto total_start = std::chrono::high_resolution_clock::now();
    auto encoder_start = total_start;
    auto decoder_start = total_start;
    
    // ===== Encoder 推理 =====
    std::vector<float> encoder_output(encoder_seq_len_ * hidden_size_, 0.0f);
    
    encoder_start = std::chrono::high_resolution_clock::now();
    int ret = run_encoder(input_ids, attention_mask, seq_len, encoder_output.data());
    auto encoder_end = std::chrono::high_resolution_clock::now();
    
    if (ret < 0) return -1;
    
    double encoder_ms = std::chrono::duration<double, std::milli>(encoder_end - encoder_start).count();
    std::cout << "[PERF] Encoder 耗时: " << encoder_ms << " ms\n";

    // ===== Decoder 推理（Zero-Copy 优化）=====
    std::vector<int64_t> decoder_input_ids = {pad_token_id_};
    std::vector<int64_t> generated_tokens;
    int out_len = 0;
    
    const float REPETITION_PENALTY = 1.2;
    const int NO_REPEAT_NGRAM_SIZE = 3;
    
    std::cout << "[DEBUG] 使用 repetition_penalty=" << REPETITION_PENALTY 
              << ", no_repeat_ngram_size=" << NO_REPEAT_NGRAM_SIZE << "\n";
    
    // ✅ Zero-Copy 优化：一次性设置固定输入（循环外）
    if (use_zero_copy_) {
        std::cout << "[INFO] 设置 Decoder 固定输入 (encoder_hidden_states & encoder_attention_mask)\n";
        
        // Input 1: encoder_hidden_states (需要类型转换)
        if (decoder_input_attrs_[1].type == RKNN_TENSOR_FLOAT16) {
            std::cout << "[DEBUG] 转换 encoder_hidden_states: FP32 → FP16\n";
            float_to_fp16_neon(encoder_output.data(),
                              (uint16_t*)decoder_input_mems_[1]->virt_addr,
                              encoder_seq_len_ * hidden_size_);
        } else {
            memcpy(decoder_input_mems_[1]->virt_addr, encoder_output.data(),
                   encoder_seq_len_ * hidden_size_ * sizeof(float));
        }
        
        ret = rknn_set_io_mem(decoder_ctx_, decoder_input_mems_[1], &decoder_input_attrs_[1]);
        if (ret < 0) {
            std::cerr << "[ERROR] 设置 Decoder encoder_hidden_states 失败\n";
            return -1;
        }
        
        // Input 2: encoder_attention_mask
        memcpy(decoder_input_mems_[2]->virt_addr, attention_mask, 
               encoder_seq_len_ * sizeof(int64_t));
        
        ret = rknn_set_io_mem(decoder_ctx_, decoder_input_mems_[2], &decoder_input_attrs_[2]);
        if (ret < 0) {
            std::cerr << "[ERROR] 设置 Decoder encoder_attention_mask 失败\n";
            return -1;
        }
        
        // Output
        ret = rknn_set_io_mem(decoder_ctx_, decoder_output_mems_[0], &decoder_output_attrs_[0]);
        if (ret < 0) {
            std::cerr << "[ERROR] 设置 Decoder 输出失败\n";
            return -1;
        }
        
        std::cout << "[INFO] ✅ Decoder 固定输入已设置（循环中只需更新 decoder_input_ids）\n";
    }
    
    decoder_start = std::chrono::high_resolution_clock::now();
    
    // ===== 解码循环 =====
    for (int step = 0; step < max_output_len && step < 30; ++step) {
        std::vector<float> logits(vocab_size_, 0.0f);
        
        // ✅ Zero-Copy 模式：run_decoder_zerocopy 只需更新 decoder_input_ids
        ret = run_decoder(decoder_input_ids.data(), decoder_input_ids.size(),
                         encoder_output.data(), attention_mask, logits.data());
        
        if (ret < 0) return -1;

        // 应用 repetition penalty
        apply_repetition_penalty(logits.data(), vocab_size_, generated_tokens, REPETITION_PENALTY);
        
        // 阻止重复的 n-gram
        block_repeated_ngrams(logits.data(), vocab_size_, generated_tokens, NO_REPEAT_NGRAM_SIZE);
        
        int next_token = argmax(logits.data(), vocab_size_);
        
        std::cout << "[DEBUG] Step " << step << ": token " << next_token;
        if (next_token == eos_token_id_) std::cout << " [EOS]";
        if (next_token == pad_token_id_) std::cout << " [PAD]";
        std::cout << "\n";
        
        output_ids[out_len++] = next_token;
        generated_tokens.push_back(next_token);

        if (next_token == eos_token_id_) {
            std::cout << "[INFO] EOS detected\n";
            break;
        }
        
        decoder_input_ids.push_back(next_token);
    }
    
    auto decoder_end = std::chrono::high_resolution_clock::now();
    auto total_end = decoder_end;
    
    double decoder_ms = std::chrono::duration<double, std::milli>(decoder_end - decoder_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    // ===== 性能统计 =====
    if (perf_stats != nullptr) {
        perf_stats->encoder_time_ms = encoder_ms;
        perf_stats->decoder_time_ms = decoder_ms;
        perf_stats->decoder_avg_time_ms = (out_len > 0) ? (decoder_ms / out_len) : 0.0;
        perf_stats->total_time_ms = total_ms;
        perf_stats->total_steps = out_len;
        perf_stats->output_tokens = out_len;
        perf_stats->zero_copy_enabled = use_zero_copy_;
        
        if (use_zero_copy_) {
            perf_stats->encoder_dma_size = encoder_input_mems_[0]->size + encoder_input_mems_[1]->size;
            perf_stats->decoder_dma_size = decoder_input_mems_[0]->size + 
                                          decoder_input_mems_[1]->size + 
                                          decoder_input_mems_[2]->size;
        }
    }

    std::cout << "\n========== 性能统计 ==========\n";
    std::cout << "[PERF] Encoder 耗时: " << encoder_ms << " ms\n";
    std::cout << "[PERF] Decoder 总耗时: " << decoder_ms << " ms\n";
    std::cout << "[PERF] Decoder 平均每步: " 
              << (out_len > 0 ? decoder_ms / out_len : 0.0) << " ms\n";
    std::cout << "[PERF] 总耗时: " << total_ms << " ms\n";
    std::cout << "[PERF] Zero-Copy: " << (use_zero_copy_ ? "启用" : "禁用") << "\n";
    std::cout << "================================\n\n";
    
    std::cout << "[INFO] 翻译完成，输出长度: " << out_len << "\n";
    std::cout << "[INFO] 输出 tokens: [";
    for (int i = 0; i < out_len; i++) {
        std::cout << output_ids[i];
        if (i < out_len - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    return out_len;
}

// ==================== 计算 argmax ====================

int HelsinkiTranslator::argmax(const float* logits, int size) {
    int max_idx = 0;
    float max_val = logits[0];
    
    for (int i = 1; i < size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// ==================== 应用 repetition penalty ====================

void HelsinkiTranslator::apply_repetition_penalty(float* logits, 
                                                  int vocab_size,
                                                  const std::vector<int64_t>& generated_tokens,
                                                  float penalty) {
    if (penalty == 1.0f) return;
    
    for (int64_t token : generated_tokens) {
        if (token >= 0 && token < vocab_size) {
            if (logits[token] > 0) {
                logits[token] /= penalty;
            } else {
                logits[token] *= penalty;
            }
        }
    }
}

// ==================== 阻止重复的 n-gram ====================

void HelsinkiTranslator::block_repeated_ngrams(float* logits, 
                                               int vocab_size,
                                               const std::vector<int64_t>& generated_tokens,
                                               int ngram_size) {
    if ((int)generated_tokens.size() < ngram_size - 1) return;
    
    std::vector<int64_t> context;
    int start_idx = generated_tokens.size() - (ngram_size - 1);
    for (size_t i = start_idx; i < generated_tokens.size(); i++) {
        context.push_back(generated_tokens[i]);
    }
    
    for (int candidate = 0; candidate < vocab_size; candidate++) {
        std::vector<int64_t> test_ngram = context;
        test_ngram.push_back(candidate);
        
        if ((int)generated_tokens.size() >= ngram_size) {
            for (size_t i = 0; i <= generated_tokens.size() - ngram_size; i++) {
                bool match = true;
                for (int j = 0; j < ngram_size; j++) {
                    if (generated_tokens[i + j] != test_ngram[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    logits[candidate] = -1e9f;
                    break;
                }
            }
        }
    }
}

// ==================== 打印模型详细信息 ====================

void HelsinkiTranslator::print_model_info() {
    std::cout << "\n[INFO] ========== 模型信息 ==========\n";
    
    rknn_sdk_version version;
    if (rknn_query(encoder_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == 0) {
        std::cout << "[INFO] Encoder SDK Version: " << version.api_version << "\n";
        std::cout << "[INFO] Encoder Driver Version: " << version.drv_version << "\n";
    }
    
    if (rknn_query(decoder_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == 0) {
        std::cout << "[INFO] Decoder SDK Version: " << version.api_version << "\n";
        std::cout << "[INFO] Decoder Driver Version: " << version.drv_version << "\n";
    }
    
    // 查询 encoder 输入输出 tensor 信息
    rknn_input_output_num io_num;
    if (rknn_query(encoder_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) == 0) {
        std::cout << "[INFO] Encoder - Inputs: " << io_num.n_input 
                  << ", Outputs: " << io_num.n_output << "\n";

        for (uint32_t i = 0; i < io_num.n_input; i++) {
            rknn_tensor_attr attr;
            attr.index = i;
            if (rknn_query(encoder_ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)) == 0) {
                std::cout << "  Input[" << i << "]: " << attr.name 
                          << ", type=" << attr.type 
                          << ", fmt=" << attr.fmt 
                          << ", dims=[";
                for (uint32_t j = 0; j < attr.n_dims; j++) {
                    std::cout << attr.dims[j];
                    if (j < attr.n_dims - 1) std::cout << ",";
                }
                std::cout << "], size=" << attr.size << " bytes\n";
            }
        }
        
        for (uint32_t i = 0; i < io_num.n_output; i++) {
            rknn_tensor_attr attr;
            attr.index = i;
            if (rknn_query(encoder_ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) == 0) {
                std::cout << "  Output[" << i << "]: " << attr.name 
                          << ", type=" << attr.type 
                          << ", fmt=" << attr.fmt 
                          << ", dims=[";
                for (uint32_t j = 0; j < attr.n_dims; j++) {
                    std::cout << attr.dims[j];
                    if (j < attr.n_dims - 1) std::cout << ",";
                }
                std::cout << "], size=" << attr.size << " bytes\n";
            }
        }
    }
    
    // 查询 decoder 输入输出 tensor 信息
    if (rknn_query(decoder_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) == 0) {
        std::cout << "[INFO] Decoder - Inputs: " << io_num.n_input 
                  << ", Outputs: " << io_num.n_output << "\n";
        
        for (uint32_t i = 0; i < io_num.n_input; i++) {
            rknn_tensor_attr attr;
            attr.index = i;
            if (rknn_query(decoder_ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)) == 0) {
                std::cout << "  Input[" << i << "]: " << attr.name 
                          << ", type=" << attr.type 
                          << ", fmt=" << attr.fmt 
                          << ", dims=[";
                for (uint32_t j = 0; j < attr.n_dims; j++) {
                    std::cout << attr.dims[j];
                    if (j < attr.n_dims - 1) std::cout << ",";
                }
                std::cout << "], size=" << attr.size << " bytes\n";
            }
        }
        
        for (uint32_t i = 0; i < io_num.n_output; i++) {
            rknn_tensor_attr attr;
            attr.index = i;
            if (rknn_query(decoder_ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) == 0) {
                std::cout << "  Output[" << i << "]: " << attr.name 
                          << ", type=" << attr.type 
                          << ", fmt=" << attr.fmt 
                          << ", dims=[";
                for (uint32_t j = 0; j < attr.n_dims; j++) {
                    std::cout << attr.dims[j];
                    if (j < attr.n_dims - 1) std::cout << ",";
                }
                std::cout << "], size=" << attr.size << " bytes\n";
            }
        }
    }
    
    std::cout << "[INFO] ==================================\n\n";
}

// ==================== 释放所有资源 ====================

void HelsinkiTranslator::release() {
    if (!initialized_) return;
    
    std::cout << "[INFO] 释放模型资源...\n";
    
    // 释放 Zero-Copy 内存
    release_zero_copy_memory();
    
    // 释放 RKNN 上下文
    if (encoder_ctx_) {
        rknn_destroy(encoder_ctx_);
        encoder_ctx_ = 0;
        std::cout << "[INFO] Encoder 上下文已释放\n";
    }
    
    if (decoder_ctx_) {
        rknn_destroy(decoder_ctx_);
        decoder_ctx_ = 0;
        std::cout << "[INFO] Decoder 上下文已释放\n";
    }
    
    initialized_ = false;
    std::cout << "[INFO] ✅ 所有资源已释放\n";
}
