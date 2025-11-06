#include <stdio.h>
#include <string.h>
#include "helsinki_api.h"

// ============================================================
// 测试程序 - 使用Helsinki Translation API
// ============================================================

void print_separator() {
    printf("========================================\n");
}

void print_usage(const char* program_name) {
    printf("Usage: %s <encoder> <decoder> <source_spm> <target_spm> <vocab> <text1> [text2] ...\n", program_name);
    printf("\nExample:\n");
    printf("  %s ./model/encoder.rknn ./model/decoder.rknn \\\n", program_name);
    printf("      ./tokenizer/source.spm ./tokenizer/target.spm \\\n");
    printf("      ./tokenizer/vocab.txt \"你好\" \"世界\"\n");
}

int main(int argc, char** argv) {
    if (argc < 7) {
        print_usage(argv[0]);
        return -1;
    }
    
    const char* encoder_path = argv[1];
    const char* decoder_path = argv[2];
    const char* source_spm   = argv[3];
    const char* target_spm   = argv[4];
    const char* vocab_txt    = argv[5];
    
    int text_count = argc - 6;
    
    print_separator();
    printf("Helsinki Translation API Test\n");
    printf("API Version: %s\n", get_api_version());
    print_separator();
    printf("\n");
    
    // ============================================================
    // 1. 初始化会话
    // ============================================================
    printf("Initializing session...\n");
    int ret = init_translation_session(
        encoder_path,
        decoder_path,
        source_spm,
        target_spm,
        vocab_txt,
        0  // verbose=0 for production
    );
    
    if (ret != 0) {
        printf("Failed to initialize!\n");
        return -1;
    }
    
    printf("\n");
    
    // ============================================================
    // 2. 执行翻译（多次调用）
    // ============================================================
    printf("Testing translations...\n");
    print_separator();
    printf("\n");
    
    char result[2048];
    int success = 0;
    int failed = 0;
    
    for (int i = 0; i < text_count; i++) {
        const char* text = argv[6 + i];
        memset(result, 0, sizeof(result));
        
        printf("[%d/%d] Input: \"%s\"\n", i + 1, text_count, text);
        
        ret = run_translation(text, result, sizeof(result));
        
        if (ret == 0) {
            printf("     Output: \"%s\" ✓\n\n", result);
            success++;
        } else {
            printf("     Output: FAILED ✗\n\n");
            failed++;
        }
    }
    
    // ============================================================
    // 3. 释放资源
    // ============================================================
    printf("\n");
    ret = release_translation_session();
    
    if (ret != 0) {
        printf("Failed to release!\n");
        return -1;
    }
    
    // ============================================================
    // 4. 打印统计
    // ============================================================
    print_separator();
    printf("Test Summary\n");
    print_separator();
    printf("Total:    %d\n", text_count);
    printf("Success:  %d\n", success);
    printf("Failed:   %d\n", failed);
    printf("Rate:     %.1f%%\n", text_count > 0 ? (100.0 * success / text_count) : 0.0);
    print_separator();
    
    return (failed == 0) ? 0 : -1;
}