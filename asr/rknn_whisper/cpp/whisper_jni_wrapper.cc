// whisper_jni_wrapper.cc
// JNI 封装层：异步流式接口

#include <jni.h>
#include <android/log.h>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

// 声明 main.cc 中的接口函数
extern "C" int init_session(const char* encoder_path, const char* decoder_path, 
                            const char* vocab_path, const char* mel_filters_path,
                            const char* task);
extern "C" int feed_audio(const float* samples, size_t count, 
                         char* result_buffer, size_t buffer_size);
extern "C" int release_session();
extern "C" const char* get_accumulated_result();
extern "C" void reset_accumulated_result();

// 可选：保留原来的一次性处理接口（用于离线文件测试）
extern "C" int main(int argc, char** argv);

#define LOG_TAG    "WhisperJNI"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

// ============================================================
// JNI 接口 1: 初始化识别会话
// ============================================================
// Java 签名:
// public static native int initSession(
//     String encoderPath, String decoderPath, String vocabPath,
//     String melFiltersPath, String language);
//
extern "C"
JNIEXPORT jint JNICALL
Java_com_example_whisper_WhisperJNI_initSession(
    JNIEnv* env,
    jclass,
    jstring jEncoderPath,
    jstring jDecoderPath,
    jstring jVocabPath,
    jstring jMelFiltersPath,
    jstring jLanguage
) {
    LOGI("═══════════════════════════════════════");
    LOGI("JNI: initSession() called");
    LOGI("═══════════════════════════════════════");
    
    const char* encoderPath = env->GetStringUTFChars(jEncoderPath, nullptr);
    const char* decoderPath = env->GetStringUTFChars(jDecoderPath, nullptr);
    const char* vocabPath = env->GetStringUTFChars(jVocabPath, nullptr);
    const char* melFiltersPath = env->GetStringUTFChars(jMelFiltersPath, nullptr);
    const char* language = env->GetStringUTFChars(jLanguage, nullptr);
    
    if (!encoderPath || !decoderPath || !vocabPath || 
        !melFiltersPath || !language) {
        LOGE("Failed to get string parameters from Java");
        
        if (encoderPath) env->ReleaseStringUTFChars(jEncoderPath, encoderPath);
        if (decoderPath) env->ReleaseStringUTFChars(jDecoderPath, decoderPath);
        if (vocabPath) env->ReleaseStringUTFChars(jVocabPath, vocabPath);
        if (melFiltersPath) env->ReleaseStringUTFChars(jMelFiltersPath, melFiltersPath);
        if (language) env->ReleaseStringUTFChars(jLanguage, language);
        
        return -1;
    }
    
    LOGI("Parameters:");
    LOGI("  Encoder:     %s", encoderPath);
    LOGI("  Decoder:     %s", decoderPath);
    LOGI("  Vocabulary:  %s", vocabPath);
    LOGI("  Mel Filters: %s", melFiltersPath);
    LOGI("  Language:    %s", language);
    
    int ret = init_session(encoderPath, decoderPath, vocabPath, 
                          melFiltersPath, language);
    
    env->ReleaseStringUTFChars(jEncoderPath, encoderPath);
    env->ReleaseStringUTFChars(jDecoderPath, decoderPath);
    env->ReleaseStringUTFChars(jVocabPath, vocabPath);
    env->ReleaseStringUTFChars(jMelFiltersPath, melFiltersPath);
    env->ReleaseStringUTFChars(jLanguage, language);
    
    if (ret == 0) {
        LOGI("JNI: Session initialized successfully ✓");
        LOGI("JNI: Background inference thread started");
    } else {
        LOGE("JNI: Session initialization failed with code %d ✗", ret);
    }
    
    LOGI("═══════════════════════════════════════\n");
    return ret;
}

// ============================================================
// JNI 接口 2: 输入音频数据（非阻塞，立即返回）
// ============================================================
// Java 签名:
// public static native void feedAudioAsync(short[] audioData);
//
extern "C"
JNIEXPORT void JNICALL
Java_com_example_whisper_WhisperJNI_feedAudioAsync(
    JNIEnv* env,
    jclass,
    jshortArray jAudioData
) {
    jshort* audioData = env->GetShortArrayElements(jAudioData, nullptr);
    if (!audioData) {
        LOGE("JNI: Failed to get audio data from Java array");
        return;
    }
    
    jsize length = env->GetArrayLength(jAudioData);
    
    // 转换 int16 → float 并归一化
    std::vector<float> floatData(length);
    for (jsize i = 0; i < length; i++) {
        floatData[i] = audioData[i] / 32768.0f;
    }
    
    // ✅ 非阻塞调用：立即返回，数据进入后台推理队列
    char dummy_buffer[1] = {0};
    feed_audio(floatData.data(), static_cast<size_t>(length), 
               dummy_buffer, sizeof(dummy_buffer));
    
    env->ReleaseShortArrayElements(jAudioData, audioData, JNI_ABORT);
}

// ============================================================
// JNI 接口 3: 尝试获取新识别结果（非阻塞）
// ============================================================
// Java 签名:
// public static native String tryGetResult();
// 返回: 新识别的文本，如果没有新结果则返回 null
//
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_whisper_WhisperJNI_tryGetResult(
    JNIEnv* env,
    jclass
) {
    // ✅ 非阻塞尝试获取结果
    char resultBuffer[2048] = {0};
    int ret = feed_audio(nullptr, 0, resultBuffer, sizeof(resultBuffer));
    
    if (ret > 0 && resultBuffer[0] != '\0') {
        return env->NewStringUTF(resultBuffer);
    }
    
    return nullptr;  // Java 端会收到 null
}

// ============================================================
// JNI 接口 4: 阻塞等待新识别结果（带超时）
// ============================================================
// Java 签名:
// public static native String getResult(int timeoutMs);
// 返回: 新识别的文本，超时或出错返回 null
//
// ⚠️ 注意：此接口需要在 main.cc 中实现 get_result_blocking()
// 这里先提供框架代码
extern "C" const char* get_result_blocking(int timeout_ms);  // 需要在 main.cc 实现

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_whisper_WhisperJNI_getResult(
    JNIEnv* env,
    jclass,
    jint timeoutMs
) {
    const char* result = get_result_blocking(timeoutMs);
    
    if (result && result[0] != '\0') {
        return env->NewStringUTF(result);
    }
    
    return nullptr;
}

// ============================================================
// JNI 接口 5: 获取累积的完整识别结果
// ============================================================
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_whisper_WhisperJNI_getAccumulatedResult(
    JNIEnv* env,
    jclass
) {
    const char* result = get_accumulated_result();
    return env->NewStringUTF(result ? result : "");
}

// ============================================================
// JNI 接口 6: 重置累积结果
// ============================================================
extern "C"
JNIEXPORT void JNICALL
Java_com_example_whisper_WhisperJNI_resetAccumulatedResult(
    JNIEnv* env,
    jclass
) {
    LOGI("JNI: Resetting accumulated result");
    reset_accumulated_result();
}

// ============================================================
// JNI 接口 7: 标记音频输入结束（用于离线文件）
// ============================================================
// Java 签名:
// public static native void finishInput();
//
extern "C" void finish_audio_input();  // 需要在 main.cc 实现

extern "C"
JNIEXPORT void JNICALL
Java_com_example_whisper_WhisperJNI_finishInput(
    JNIEnv* env,
    jclass
) {
    LOGI("JNI: Marking input as finished");
    finish_audio_input();
}

// ============================================================
// JNI 接口 8: 等待所有推理完成
// ============================================================
// Java 签名:
// public static native void waitForCompletion();
//
extern "C" void wait_for_inference_completion();  // 需要在 main.cc 实现

extern "C"
JNIEXPORT void JNICALL
Java_com_example_whisper_WhisperJNI_waitForCompletion(
    JNIEnv* env,
    jclass
) {
    LOGI("JNI: Waiting for inference completion...");
    wait_for_inference_completion();
    LOGI("JNI: All inferences completed");
}

// ============================================================
// JNI 接口 9: 获取统计信息
// ============================================================
// Java 签名:
// public static native int getInferenceCount();
//
extern "C" int get_inference_count();  // 需要在 main.cc 实现

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_whisper_WhisperJNI_getInferenceCount(
    JNIEnv* env,
    jclass
) {
    return get_inference_count();
}

// ============================================================
// JNI 接口 10: 获取平均推理时间
// ============================================================
// Java 签名:
// public static native double getAverageInferenceTime();
//
extern "C" double get_average_inference_time();  // 需要在 main.cc 实现

extern "C"
JNIEXPORT jdouble JNICALL
Java_com_example_whisper_WhisperJNI_getAverageInferenceTime(
    JNIEnv* env,
    jclass
) {
    return get_average_inference_time();
}

// ============================================================
// JNI 接口 11: 释放会话资源
// ============================================================
extern "C"
JNIEXPORT jint JNICALL
Java_com_example_whisper_WhisperJNI_releaseSession(
    JNIEnv* env,
    jclass
) {
    LOGI("═══════════════════════════════════════");
    LOGI("JNI: releaseSession() called");
    LOGI("═══════════════════════════════════════");
    
    int ret = release_session();
    
    if (ret == 0) {
        LOGI("JNI: Session released successfully ✓");
    } else {
        LOGE("JNI: Session release failed with code %d ✗", ret);
    }
    
    LOGI("═══════════════════════════════════════\n");
    return ret;
}

// ============================================================
// （可选）JNI 接口 12: 一次性处理音频文件（保留用于测试）
// ============================================================
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_whisper_WhisperJNI_runWhisperOnce(
    JNIEnv* env,
    jclass,
    jstring jenc,
    jstring jdec,
    jstring jtask,
    jstring jaud,
    jstring jvocab,
    jstring jmel
) {
    LOGI("JNI: runWhisperOnce() called (legacy interface)");

    const char* enc   = env->GetStringUTFChars(jenc, nullptr);
    const char* dec   = env->GetStringUTFChars(jdec, nullptr);
    const char* task  = env->GetStringUTFChars(jtask, nullptr);
    const char* aud   = env->GetStringUTFChars(jaud, nullptr);
    const char* vocab = env->GetStringUTFChars(jvocab, nullptr);
    const char* mel   = env->GetStringUTFChars(jmel, nullptr);
    
    if (!enc || !dec || !task || !aud || !vocab || !mel) {
        LOGE("One of GetStringUTFChars returned null");
        if (enc)   env->ReleaseStringUTFChars(jenc,  enc);
        if (dec)   env->ReleaseStringUTFChars(jdec,  dec);
        if (task)  env->ReleaseStringUTFChars(jtask, task);
        if (aud)   env->ReleaseStringUTFChars(jaud,  aud);
        if (vocab) env->ReleaseStringUTFChars(jvocab, vocab);
        if (mel)   env->ReleaseStringUTFChars(jmel,   mel);
        return env->NewStringUTF("");
    }

    int   argc = 7;
    char* argv[7];
    argv[0] = (char*)"rknn_whisper_demo";
    argv[1] = const_cast<char*>(enc);
    argv[2] = const_cast<char*>(dec);
    argv[3] = const_cast<char*>(task);
    argv[4] = const_cast<char*>(aud);
    argv[5] = const_cast<char*>(vocab);
    argv[6] = const_cast<char*>(mel);

    LOGI("Calling main() with:");
    LOGI("  enc=%s", enc);
    LOGI("  dec=%s", dec);
    LOGI("  task=%s", task);
    LOGI("  aud=%s", aud);
    LOGI("  vocab=%s", vocab);
    LOGI("  mel=%s", mel);
    
    int ret = main(argc, argv);
    LOGI("native main() returned %d", ret);

    const char* out = get_accumulated_result();
    if (out && out[0]) {
        LOGI("Result length: %zu", strlen(out));
    } else {
        LOGI("Result is empty");
    }

    env->ReleaseStringUTFChars(jenc,  enc);
    env->ReleaseStringUTFChars(jdec,  dec);
    env->ReleaseStringUTFChars(jtask, task);
    env->ReleaseStringUTFChars(jaud,  aud);
    env->ReleaseStringUTFChars(jvocab, vocab);
    env->ReleaseStringUTFChars(jmel,   mel);

    return env->NewStringUTF(out ? out : "");
}