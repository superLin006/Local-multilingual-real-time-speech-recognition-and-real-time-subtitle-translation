package com.example.helsinki;

/**
 * Helsinki Translation JNI Interface
 * 实时ASR翻译场景专用接口
 * 
 * 使用流程:
 * 1. 应用启动时调用 initSession() 初始化一次
 * 2. 每次需要翻译时调用 translate()，可以高频调用
 * 3. 应用退出时调用 releaseSession() 释放一次
 * 
 * 示例:
 * <pre>
 * // 1. 初始化（应用启动时）
 * int ret = HelsinkiJNI.initSession(
 *     "/path/to/encoder.rknn",
 *     "/path/to/decoder.rknn",
 *     "/path/to/source.spm",
 *     "/path/to/target.spm",
 *     "/path/to/vocab.txt",
 *     false  // verbose mode
 * );
 * 
 * // 2. 翻译（可多次调用）
 * String result1 = HelsinkiJNI.translate("你好");
 * String result2 = HelsinkiJNI.translate("世界");
 * String result3 = HelsinkiJNI.translate("今天天气很好");
 * 
 * // 3. 释放（应用退出时）
 * HelsinkiJNI.releaseSession();
 * </pre>
 */
public class HelsinkiJNI {
    
    // 加载native库
    static {
        System.loadLibrary("rknn_helsinki_demo");
    }
    
    // ============================================================
    // 核心接口
    // ============================================================
    
    /**
     * 初始化翻译会话
     * 
     * @param encoderPath  Encoder模型路径 (.rknn文件)
     * @param decoderPath  Decoder模型路径 (.rknn文件)
     * @param sourceSpm    Source tokenizer文件 (.spm文件)
     * @param targetSpm    Target tokenizer文件 (.spm文件)
     * @param vocabTxt     词汇表文件 (.txt文件)
     * @param verbose      是否开启详细日志
     * @return 0=成功, -1=失败
     * 
     * 注意:
     * - 应该在应用启动时调用一次
     * - 重复调用会返回警告但不会出错
     * - 此方法不是线程安全的，应在主线程调用
     */
    public static native int initSession(
        String encoderPath,
        String decoderPath,
        String sourceSpm,
        String targetSpm,
        String vocabTxt,
        boolean verbose
    );
    
    /**
     * 执行翻译（核心接口，支持高频调用）
     * 
     * @param text 输入文本 (UTF-8编码)
     * @return 翻译结果，失败返回null
     * 
     * 性能:
     * - 首次调用: ~100-200ms (模型warmup)
     * - 后续调用: ~50-100ms (取决于文本长度)
     * 
     * 线程安全: 是 (内部使用mutex保护)
     * 
     * 适用场景:
     * - 实时ASR翻译
     * - 可以每秒调用多次
     * - 单次失败不影响后续调用
     * 
     * 注意:
     * - 必须先调用 initSession()
     */
    public static native String translate(String text);
    
    /**
     * 释放会话资源
     * 
     * @return 0=成功
     * 
     * 注意:
     * - 应该在应用退出时调用一次
     * - 调用后必须重新init才能使用翻译功能
     * - 此方法不是线程安全的，应在主线程调用
     */
    public static native int releaseSession();
    
    // ============================================================
    // 辅助接口
    // ============================================================
    
    /**
     * 查询会话是否已初始化
     * 
     * @return true=已初始化, false=未初始化
     * 
     * 线程安全: 是
     * 
     * 用途:
     * - 可以在调用translate()前检查状态
     * - 防止未初始化时调用导致错误
     */
    public static native boolean isInitialized();
    
    /**
     * 设置日志模式
     * 
     * @param verbose true=开启详细日志, false=关闭详细日志
     * 
     * 用途:
     * - 开发阶段: verbose=true 便于调试
     * - 生产环境: verbose=false 减少日志开销
     * 
     * 注意:
     * - 可以在运行时动态调用
     * - 不影响错误日志的输出
     */
    public static native void setVerboseMode(boolean verbose);
    
    /**
     * 获取API版本
     * 
     * @return 版本字符串 (例如: "1.0.0")
     */
    public static native String getApiVersion();
    
    // ============================================================
    // 便捷方法（Java层封装）
    // ============================================================
    
    /**
     * 安全的翻译方法（带状态检查）
     * 
     * @param text 输入文本
     * @return 翻译结果，失败或未初始化返回null
     */
    public static String translateSafe(String text) {
        if (!isInitialized()) {
            System.err.println("[ERROR] Session not initialized!");
            return null;
        }
        
        if (text == null || text.isEmpty()) {
            System.err.println("[ERROR] Input text is null or empty!");
            return null;
        }
        
        return translate(text);
    }
}