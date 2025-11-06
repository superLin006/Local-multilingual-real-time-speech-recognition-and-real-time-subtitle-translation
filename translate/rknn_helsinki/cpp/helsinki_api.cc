#include "helsinki_api.h"
#include "helsinki.h"
#include "sp_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <pthread.h>

// ============================================================
// 版本信息
// ============================================================
#define API_VERSION "1.0.0"

// ============================================================
// 全局变量 - 用于存储会话状态
// ============================================================
static HelsinkiTranslator* g_translator = nullptr;
static SPTokenizer* g_source_tokenizer = nullptr;
static SPTokenizer* g_target_tokenizer = nullptr;
static bool g_initialized = false;
static pthread_mutex_t g_translation_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool g_verbose_mode = false;

// ============================================================
// 日志宏
// ============================================================
#define LOG_INFO(fmt, ...) \
    do { if (g_verbose_mode) printf("[INFO] " fmt "\n", ##__VA_ARGS__); } while(0)

#define LOG_ERROR(fmt, ...) \
    printf("[ERROR] " fmt "\n", ##__VA_ARGS__)

#define LOG_WARN(fmt, ...) \
    printf("[WARN] " fmt "\n", ##__VA_ARGS__)

// ============================================================
// API 实现
// ============================================================

extern "C" int init_translation_session(
    const char* encoder_path,
    const char* decoder_path,
    const char* source_spm,
    const char* target_spm,
    const char* vocab_txt,
    int verbose
) {
    if (g_initialized) {
        LOG_WARN("Session already initialized!");
        return 0;
    }
    
    g_verbose_mode = (verbose != 0);
    
    printf("========================================\n");
    printf("Initializing Translation Session...\n");
    printf("========================================\n");
    
    printf("Configuration:\n");
    printf("  Encoder:      %s\n", encoder_path);
    printf("  Decoder:      %s\n", decoder_path);
    printf("  Source SPM:   %s\n", source_spm);
    printf("  Target SPM:   %s\n", target_spm);
    printf("  Vocabulary:   %s\n", vocab_txt);
    printf("  Verbose Mode: %s\n", verbose ? "ON" : "OFF");
    printf("  API Version:  %s\n", API_VERSION);
    printf("\n");
    
    // 创建对象
    g_source_tokenizer = new SPTokenizer();
    g_target_tokenizer = new SPTokenizer();
    g_translator = new HelsinkiTranslator();
    
    if (!g_source_tokenizer || !g_target_tokenizer || !g_translator) {
        LOG_ERROR("Memory allocation failed!");
        goto cleanup_on_error;
    }
    
    // 初始化source tokenizer
    printf("[INFO] Loading source tokenizer...\n");
    if (!g_source_tokenizer->load_model(source_spm, vocab_txt)) {
        LOG_ERROR("Failed to load source tokenizer!");
        goto cleanup_on_error;
    }
    printf("   ✓ Source tokenizer loaded\n");
    
    // 初始化target tokenizer
    printf("[INFO] Loading target tokenizer...\n");
    if (!g_target_tokenizer->load_model(target_spm, vocab_txt)) {
        LOG_ERROR("Failed to load target tokenizer!");
        goto cleanup_on_error;
    }
    printf("   ✓ Target tokenizer loaded\n");
    
    // 初始化translator
    printf("[INFO] Loading translation models...\n");
    if (g_translator->init(encoder_path, decoder_path) != 0) {
        LOG_ERROR("Failed to initialize translator!");
        goto cleanup_on_error;
    }
    printf("   ✓ Translation models loaded\n");
    
    g_initialized = true;
    
    printf("\n========================================\n");
    printf("Session initialized successfully!\n");
    printf("========================================\n\n");
    
    return 0;

cleanup_on_error:
    if (g_source_tokenizer) {
        delete g_source_tokenizer;
        g_source_tokenizer = nullptr;
    }
    if (g_target_tokenizer) {
        delete g_target_tokenizer;
        g_target_tokenizer = nullptr;
    }
    if (g_translator) {
        delete g_translator;
        g_translator = nullptr;
    }
    
    LOG_ERROR("Session initialization failed!");
    return -1;
}

extern "C" int run_translation(
    const char* input_text,
    char* output_buffer,
    size_t buffer_size
) {
    // 参数检查（不加锁）
    if (!input_text || !output_buffer || buffer_size == 0) {
        LOG_ERROR("Invalid parameters!");
        return -1;
    }
    
    // 加锁保护（防止多线程同时调用）
    pthread_mutex_lock(&g_translation_mutex);
    
    // 初始化检查
    if (!g_initialized) {
        LOG_ERROR("Session not initialized! Call init_translation_session() first.");
        pthread_mutex_unlock(&g_translation_mutex);
        return -1;
    }
    
    LOG_INFO(">>> Translation: \"%s\"", input_text);
    
    int ret = 0;
    
    // ⭐ 关键修改：在goto之前声明所有变量
    const int MAX_SEQ_LEN = 64;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> padded_input_ids;
    std::vector<int64_t> attention_mask(MAX_SEQ_LEN, 0);
    int64_t output_ids[64] = {0};
    int output_len = 0;
    
    // Step 1: Tokenize input
    try {
        input_ids = g_source_tokenizer->encode(input_text, true);
        LOG_INFO("Tokenized: %zu tokens", input_ids.size());
    } catch (...) {
        LOG_ERROR("Tokenization failed!");
        ret = -1;
        goto unlock_and_return;
    }
    
    // Step 2: Pad sequences
    try {
        padded_input_ids = g_source_tokenizer->pad_sequence(
            input_ids, MAX_SEQ_LEN, g_source_tokenizer->get_pad_token_id()
        );
        
        for (size_t i = 0; i < input_ids.size() && i < MAX_SEQ_LEN; i++) {
            attention_mask[i] = 1;
        }
    } catch (...) {
        LOG_ERROR("Padding failed!");
        ret = -1;
        goto unlock_and_return;
    }
    
    // Step 3: Translate
    try {
        output_len = g_translator->translate(
            padded_input_ids.data(),
            attention_mask.data(),
            MAX_SEQ_LEN,
            output_ids,
            64
        );
        
        if (output_len <= 0) {
            LOG_ERROR("Translation inference failed!");
            ret = -1;
            goto unlock_and_return;
        }
        
        LOG_INFO("Generated %d tokens", output_len);
    } catch (...) {
        LOG_ERROR("Translation exception!");
        ret = -1;
        goto unlock_and_return;
    }
    
    // Step 4: Decode output
    try {
        std::vector<int64_t> output_token_ids(output_ids, output_ids + output_len);
        std::string translation = g_target_tokenizer->decode(output_token_ids, true);
        
        // Copy result to output buffer
        if (translation.length() >= buffer_size) {
            LOG_WARN("Output truncated (buffer too small: %zu < %zu)", 
                    buffer_size, translation.length() + 1);
            strncpy(output_buffer, translation.c_str(), buffer_size - 1);
            output_buffer[buffer_size - 1] = '\0';
        } else {
            strcpy(output_buffer, translation.c_str());
        }
        
        LOG_INFO("<<< Result: \"%s\"", output_buffer);
    } catch (...) {
        LOG_ERROR("Decoding failed!");
        ret = -1;
        goto unlock_and_return;
    }
    
unlock_and_return:
    pthread_mutex_unlock(&g_translation_mutex);
    return ret;
}

extern "C" int release_translation_session() {
    if (!g_initialized) {
        LOG_WARN("Session not initialized, nothing to release.");
        return 0;
    }
    
    printf("========================================\n");
    printf("Releasing Translation Session...\n");
    printf("========================================\n");
    
    // 释放translator
    if (g_translator) {
        printf("[INFO] Releasing translator...\n");
        g_translator->release();
        delete g_translator;
        g_translator = nullptr;
        printf("   ✓ Translator released\n");
    }
    
    // 释放tokenizers
    if (g_source_tokenizer) {
        delete g_source_tokenizer;
        g_source_tokenizer = nullptr;
        printf("   ✓ Source tokenizer released\n");
    }
    
    if (g_target_tokenizer) {
        delete g_target_tokenizer;
        g_target_tokenizer = nullptr;
        printf("   ✓ Target tokenizer released\n");
    }
    
    g_initialized = false;
    
    printf("\n========================================\n");
    printf("Session released successfully!\n");
    printf("========================================\n\n");
    
    return 0;
}

extern "C" int is_session_initialized() {
    return g_initialized ? 1 : 0;
}

extern "C" void set_verbose_mode(int verbose) {
    g_verbose_mode = (verbose != 0);
    printf("[INFO] Verbose mode %s\n", g_verbose_mode ? "ON" : "OFF");
}

extern "C" const char* get_api_version() {
    return API_VERSION;
}