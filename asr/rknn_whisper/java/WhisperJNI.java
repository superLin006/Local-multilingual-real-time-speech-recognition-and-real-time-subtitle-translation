package com.example.whisper;

public class WhisperJNI {
    static {
        System.loadLibrary("rknn_whisper_demo");
    }

    // 初始化会话
    public static native int initSession(
        String encoderPath, String decoderPath, String vocabPath,
        String melFiltersPath, String language);

    // ✅ 非阻塞喂入音频（立即返回）
    public static native void feedAudioAsync(short[] audioData);

    // ✅ 非阻塞尝试获取结果（立即返回，无结果返回null）
    public static native String tryGetResult();

    // ✅ 阻塞获取结果（等待直到有结果或超时，超时返回null）
    public static native String getResult(int timeoutMs);

    // 获取累积的完整结果
    public static native String getAccumulatedResult();

    // 重置累积结果
    public static native void resetAccumulatedResult();

    // ✅ 标记输入结束（用于离线文件处理）
    public static native void finishInput();

    // ✅ 等待所有推理完成
    public static native void waitForCompletion();

    // ✅ 获取统计信息
    public static native int getInferenceCount();
    public static native double getAverageInferenceTime();

    // 释放会话
    public static native int releaseSession();

    // 保留的一次性处理接口
    public static native String runWhisperOnce(
        String encoderPath, String decoderPath, String task,
        String audioPath, String vocabPath, String melFiltersPath);
}