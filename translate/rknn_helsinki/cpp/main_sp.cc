#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <pthread.h>
#include "helsinki.h"
#include "sp_tokenizer.h"

// ============================================================
// 全局变量 - 用于存储会话状态
// ============================================================
static HelsinkiTranslator* g_translator = nullptr;
static SPTokenizer* g_source_tokenizer = nullptr;
static SPTokenizer* g_target_tokenizer = nullptr;
static bool g_initialized = false;
static pthread_mutex_t g_translation_mutex = PTHREAD_MUTEX_INITIALIZER;  // 线程锁
static bool g_verbose_mode = false;  // 详细日志模式（调试用）

// ============================================================
// 辅助函数：日志控制
// ============================================================
#define LOG_INFO(fmt, ...) \
    do { if (g_verbose_mode) printf("[INFO] " fmt "\n", ##__VA_ARGS__); } while(0)

#define LOG_ERROR(fmt, ...) \
    printf("[ERROR] " fmt "\n", ##__VA_ARGS__)

#define LOG_WARN(fmt, ...) \
    printf("[WARN] " fmt "\n", ##__VA_ARGS__)

// ============================================================
// 函数1: 模型初始化
// ============================================================
// 功能: 初始化translator和两个tokenizers
// 参数:
//   encoder_path  - encoder模型路径
//   decoder_path  - decoder模型路径
//   source_spm    - source tokenizer的.spm文件
//   target_spm    - target tokenizer的.spm文件
//   vocab_txt     - 词汇表文件
//   verbose       - 是否开启详细日志（0=关闭，1=开启）
// 返回值:
//   0  - 成功
//   -1 - 失败
// 注意: 此函数不需要加锁，应该在主线程初始化阶段调用
// ============================================================
int init_translation_session(
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

// ============================================================
// 函数2: 执行翻译推理（线程安全，适合实时调用）
// ============================================================
// 功能: 对输入文本进行翻译
// 参数:
//   input_text    - 输入文本
//   output_buffer - 输出缓冲区（用于存储翻译结果）
//   buffer_size   - 输出缓冲区大小
// 返回值:
//   0  - 成功
//   -1 - 失败
// 线程安全: 是（内部使用mutex保护）
// 适用场景: 实时ASR翻译，高频调用
// ============================================================
int run_translation(
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
    
    // ===== 将所有变量声明提前到这里，避免goto跳过初始化 =====
    int ret = 0;
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
        
        // 重新初始化 attention_mask
        attention_mask.assign(MAX_SEQ_LEN, 0);
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

// ============================================================
// 函数3: 释放资源
// ============================================================
// 功能: 释放所有已分配的资源
// 返回值:
//   0  - 成功
// 注意: 此函数不需要加锁，应该在主线程退出阶段调用
// ============================================================
int release_translation_session() {
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

// ============================================================
// 辅助函数4: 查询会话状态
// ============================================================
// 功能: 检查会话是否已初始化
// 返回值:
//   1  - 已初始化
//   0  - 未初始化
// 线程安全: 是（只读操作）
// ============================================================
int is_session_initialized() {
    return g_initialized ? 1 : 0;
}

// ============================================================
// 辅助函数5: 设置日志模式
// ============================================================
// 功能: 动态调整日志详细程度
// 参数:
//   verbose - 0=关闭详细日志，1=开启详细日志
// ============================================================
void set_verbose_mode(int verbose) {
    g_verbose_mode = (verbose != 0);
    printf("[INFO] Verbose mode %s\n", g_verbose_mode ? "ON" : "OFF");
}

// ============================================================
// 辅助函数
// ============================================================
void print_separator() {
    printf("========================================\n");
}

void print_usage(const char* program_name) {
    printf("Usage: %s <encoder_path> <decoder_path> <source_spm> <target_spm> <vocab_txt> <text1> [text2] ...\n", program_name);
    printf("\nParameters:\n");
    printf("  encoder_path  - Path to encoder RKNN model file\n");
    printf("  decoder_path  - Path to decoder RKNN model file\n");
    printf("  source_spm    - Path to source tokenizer .spm file\n");
    printf("  target_spm    - Path to target tokenizer .spm file\n");
    printf("  vocab_txt     - Path to vocabulary .txt file\n");
    printf("  text1, text2  - One or more texts to translate\n");
    printf("\nExample:\n");
    printf("  %s ./model/encoder.rknn ./model/decoder.rknn \\\n", program_name);
    printf("      ./tokenizer/source.spm ./tokenizer/target.spm \\\n");
    printf("      ./tokenizer/vocab.txt \"你好\" \"世界\" \"测试\"\n");
}

// ============================================================
// 主函数 - 模拟实时ASR翻译场景
// ============================================================
int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc < 7) {
        print_usage(argv[0]);
        return -1;
    }
    
    // 解析命令行参数
    const char* encoder_path = argv[1];
    const char* decoder_path = argv[2];
    const char* source_spm   = argv[3];
    const char* target_spm   = argv[4];
    const char* vocab_txt    = argv[5];
    
    int text_count = argc - 6;
    
    print_separator();
    printf("Helsinki Real-Time Translation Demo\n");
    printf("Simulating ASR → Translation Pipeline\n");
    print_separator();
    printf("\n");
    
    printf("Simulation: Will process %d ASR outputs\n\n", text_count);
    
    int ret = 0;
    
    // ========================================
    // 阶段1: 应用启动 - 初始化模型（仅一次）
    // ========================================
    printf("\n[PHASE 1] Application Startup\n");
    print_separator();
    
    ret = init_translation_session(
        encoder_path,
        decoder_path,
        source_spm,
        target_spm,
        vocab_txt,
        0  // verbose=0: 关闭详细日志（实时场景推荐）
    );
    
    if (ret != 0) {
        LOG_ERROR("Initialization failed!");
        return -1;
    }
    
    printf("Status: Ready for real-time translation\n\n");
    
    // ========================================
    // 阶段2: 运行时 - 持续接收ASR并翻译
    // ========================================
    printf("\n[PHASE 2] Real-Time Translation Loop\n");
    print_separator();
    printf("Simulating continuous ASR outputs...\n\n");
    
    char translation_result[2048];
    int success_count = 0;
    int failed_count = 0;
    
    for (int i = 0; i < text_count; i++) {
        const char* asr_output = argv[6 + i];
        
        printf("ASR Output #%d: \"%s\"\n", i + 1, asr_output);
        
        memset(translation_result, 0, sizeof(translation_result));
        
        // 模拟：ASR结果出来后立即翻译
        ret = run_translation(asr_output, translation_result, sizeof(translation_result));
        
        if (ret == 0) {
            success_count++;
            printf("  → Translation: \"%s\" ✓\n", translation_result);
        } else {
            failed_count++;
            printf("  → Translation: FAILED ✗\n");
        }
        
        printf("\n");
    }
    
    // ========================================
    // 阶段3: 应用退出 - 释放资源（仅一次）
    // ========================================
    printf("\n[PHASE 3] Application Shutdown\n");
    print_separator();
    
    ret = release_translation_session();
    
    if (ret != 0) {
        LOG_ERROR("Resource release failed!");
        return -1;
    }
    
    // 打印统计信息
    print_separator();
    printf("Real-Time Translation Summary\n");
    printf("========================================\n");
    printf("Total ASR Outputs:   %d\n", text_count);
    printf("Successful:          %d\n", success_count);
    printf("Failed:              %d\n", failed_count);
    printf("Success Rate:        %.1f%%\n", 
           text_count > 0 ? (100.0 * success_count / text_count) : 0.0);
    print_separator();
    printf("Demo completed!\n");
    print_separator();
    
    return (failed_count == 0) ? 0 : -1;
}