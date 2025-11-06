#include <jni.h>
#include <android/log.h>
#include <string.h>
#include "helsinki_api.h"

// ============================================================
// JNI 日志宏
// ============================================================
#define JNI_TAG    "HelsinkiJNI"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,  JNI_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, JNI_TAG, __VA_ARGS__)
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,  JNI_TAG, __VA_ARGS__)

// ============================================================
// JNI 辅助函数
// ============================================================

// 获取Java字符串
static const char* get_java_string(JNIEnv* env, jstring jstr) {
    if (jstr == nullptr) return nullptr;
    return env->GetStringUTFChars(jstr, nullptr);
}

// 释放Java字符串
static void release_java_string(JNIEnv* env, jstring jstr, const char* cstr) {
    if (jstr && cstr) {
        env->ReleaseStringUTFChars(jstr, cstr);
    }
}

// ============================================================
// JNI 接口实现
// ============================================================

// ============================================================
// 1. 初始化翻译会话
// Java签名:
// public static native int initSession(
//     String encoderPath, String decoderPath,
//     String sourceSpm, String targetSpm,
//     String vocabTxt, boolean verbose);
// ============================================================
extern "C"
JNIEXPORT jint JNICALL
Java_com_example_helsinki_HelsinkiJNI_initSession(
    JNIEnv* env,
    jclass clazz,
    jstring jEncoderPath,
    jstring jDecoderPath,
    jstring jSourceSpm,
    jstring jTargetSpm,
    jstring jVocabTxt,
    jboolean jVerbose
) {
    LOGI("========================================");
    LOGI("JNI: initSession() called");
    LOGI("========================================");
    
    // 获取所有参数
    const char* encoder_path = get_java_string(env, jEncoderPath);
    const char* decoder_path = get_java_string(env, jDecoderPath);
    const char* source_spm   = get_java_string(env, jSourceSpm);
    const char* target_spm   = get_java_string(env, jTargetSpm);
    const char* vocab_txt    = get_java_string(env, jVocabTxt);
    
    // 参数检查
    if (!encoder_path || !decoder_path || !source_spm || !target_spm || !vocab_txt) {
        LOGE("One or more parameters are null!");
        
        release_java_string(env, jEncoderPath, encoder_path);
        release_java_string(env, jDecoderPath, decoder_path);
        release_java_string(env, jSourceSpm, source_spm);
        release_java_string(env, jTargetSpm, target_spm);
        release_java_string(env, jVocabTxt, vocab_txt);
        
        return -1;
    }
    
    LOGI("Parameters:");
    LOGI("  Encoder:    %s", encoder_path);
    LOGI("  Decoder:    %s", decoder_path);
    LOGI("  Source SPM: %s", source_spm);
    LOGI("  Target SPM: %s", target_spm);
    LOGI("  Vocab:      %s", vocab_txt);
    LOGI("  Verbose:    %s", jVerbose ? "true" : "false");
    
    // 调用API
    int ret = init_translation_session(
        encoder_path,
        decoder_path,
        source_spm,
        target_spm,
        vocab_txt,
        jVerbose ? 1 : 0
    );
    
    // 释放字符串
    release_java_string(env, jEncoderPath, encoder_path);
    release_java_string(env, jDecoderPath, decoder_path);
    release_java_string(env, jSourceSpm, source_spm);
    release_java_string(env, jTargetSpm, target_spm);
    release_java_string(env, jVocabTxt, vocab_txt);
    
    if (ret == 0) {
        LOGI("JNI: Session initialized successfully ✓");
    } else {
        LOGE("JNI: Session initialization failed ✗");
    }
    
    LOGI("========================================\n");
    
    return ret;
}

// ============================================================
// 2. 执行翻译
// Java签名:
// public static native String translate(String text);
// ============================================================
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_helsinki_HelsinkiJNI_translate(
    JNIEnv* env,
    jclass clazz,
    jstring jText
) {
    // 参数检查
    if (jText == nullptr) {
        LOGE("JNI: Input text is null!");
        return nullptr;
    }
    
    // 获取输入文本
    const char* input_text = get_java_string(env, jText);
    if (!input_text) {
        LOGE("JNI: Failed to get input text!");
        return nullptr;
    }
    
    // 准备输出缓冲区
    char output_buffer[2048] = {0};
    
    // 调用翻译API
    int ret = run_translation(input_text, output_buffer, sizeof(output_buffer));
    
    // 释放输入字符串
    release_java_string(env, jText, input_text);
    
    // 检查结果
    if (ret != 0) {
        LOGE("JNI: Translation failed!");
        return nullptr;
    }
    
    // 返回Java字符串
    return env->NewStringUTF(output_buffer);
}

// ============================================================
// 3. 释放会话
// Java签名:
// public static native int releaseSession();
// ============================================================
extern "C"
JNIEXPORT jint JNICALL
Java_com_example_helsinki_HelsinkiJNI_releaseSession(
    JNIEnv* env,
    jclass clazz
) {
    LOGI("========================================");
    LOGI("JNI: releaseSession() called");
    LOGI("========================================");
    
    int ret = release_translation_session();
    
    if (ret == 0) {
        LOGI("JNI: Session released successfully ✓");
    } else {
        LOGE("JNI: Session release failed ✗");
    }
    
    LOGI("========================================\n");
    
    return ret;
}

// ============================================================
// 4. 查询会话状态
// Java签名:
// public static native boolean isInitialized();
// ============================================================
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_helsinki_HelsinkiJNI_isInitialized(
    JNIEnv* env,
    jclass clazz
) {
    int initialized = is_session_initialized();
    return (initialized != 0) ? JNI_TRUE : JNI_FALSE;
}

// ============================================================
// 5. 设置日志模式
// Java签名:
// public static native void setVerboseMode(boolean verbose);
// ============================================================
extern "C"
JNIEXPORT void JNICALL
Java_com_example_helsinki_HelsinkiJNI_setVerboseMode(
    JNIEnv* env,
    jclass clazz,
    jboolean jVerbose
) {
    LOGI("JNI: setVerboseMode(%s)", jVerbose ? "true" : "false");
    set_verbose_mode(jVerbose ? 1 : 0);
}

// ============================================================
// 6. 获取API版本
// Java签名:
// public static native String getApiVersion();
// ============================================================
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_helsinki_HelsinkiJNI_getApiVersion(
    JNIEnv* env,
    jclass clazz
) {
    const char* version = get_api_version();
    return env->NewStringUTF(version);
}