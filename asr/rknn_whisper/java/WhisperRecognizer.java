// WhisperRecognizer.java
package com.example.whisper;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

/**
 * Whisper 实时语音识别器（异步版本）
 * 封装麦克风采集和识别流程
 */
public class WhisperRecognizer {
    
    private static final String TAG = "WhisperRecognizer";
    
    // 音频配置
    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int BUFFER_SIZE_PER_READ = 1600;  // 0.1秒 (100ms)
    
    private AudioRecord audioRecord;
    private Thread recordThread;           // 录音线程
    private Thread resultThread;           // ✅ 新增：结果收集线程
    private volatile boolean isRecording = false;
    private volatile boolean isCollectingResults = false;  // ✅ 新增
    private RecognitionCallback callback;
    
    /**
     * 识别结果回调接口
     */
    public interface RecognitionCallback {
        /**
         * 当有新的识别文本时调用
         * @param text 新识别的文本片段
         */
        void onNewText(String text);
        
        /**
         * 当发生错误时调用
         * @param error 错误信息
         */
        void onError(String error);
        
        /**
         * ✅ 新增：当识别完成时调用（可选实现）
         */
        default void onComplete() {}
    }
    
    /**
     * 初始化识别器
     * 
     * @param modelBasePath 模型文件所在目录
     * @param language 语言代码（"en", "zh", "ja" 等）
     * @return true=成功, false=失败
     */
    public boolean initialize(String modelBasePath, String language) {
        Log.i(TAG, "Initializing Whisper recognizer (Async Mode)...");
        Log.i(TAG, "Model path: " + modelBasePath);
        Log.i(TAG, "Language: " + language);
        
        // 确保路径以 / 结尾
        if (!modelBasePath.endsWith("/")) {
            modelBasePath += "/";
        }
        
        String encoderPath = modelBasePath + "encoder.rknn";
        String decoderPath = modelBasePath + "decoder.rknn";
        String vocabPath = modelBasePath + "vocab.txt";
        String melFiltersPath = modelBasePath + "mel_80_filters.txt";
        
        // 调用 JNI 初始化（会启动后台推理线程）
        int ret = WhisperJNI.initSession(
            encoderPath,
            decoderPath,
            vocabPath,
            melFiltersPath,
            language
        );
        
        if (ret == 0) {
            Log.i(TAG, "✓ Whisper session initialized (inference thread started)");
            return true;
        } else {
            Log.e(TAG, "✗ Failed to initialize Whisper session, error code: " + ret);
            return false;
        }
    }
    
    /**
     * 开始实时识别
     * 
     * @param callback 识别结果回调
     * @return true=成功启动, false=失败
     */
    public boolean startRecognition(RecognitionCallback callback) {
        if (isRecording) {
            Log.w(TAG, "Recognition already started");
            return false;
        }
        
        this.callback = callback;
        
        // 计算缓冲区大小
        int minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT
        );
        
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(TAG, "Failed to get minimum buffer size");
            if (callback != null) {
                callback.onError("Failed to initialize audio recording");
            }
            return false;
        }
        
        // 使用双倍缓冲区防止溢出
        int bufferSize = Math.max(minBufferSize * 2, BUFFER_SIZE_PER_READ * 4);
        
        Log.i(TAG, "AudioRecord buffer size: " + bufferSize + " bytes");
        
        try {
            audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            );
            
            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord initialization failed");
                if (callback != null) {
                    callback.onError("Failed to initialize audio recording");
                }
                return false;
            }
            
            audioRecord.startRecording();
            isRecording = true;
            isCollectingResults = true;
            
            // ✅ 启动结果收集线程（优先启动）
            resultThread = new Thread(this::resultCollectionLoop, "WhisperResultThread");
            resultThread.start();
            
            // 启动录音线程
            recordThread = new Thread(this::recordingLoop, "WhisperRecordThread");
            recordThread.start();
            
            Log.i(TAG, "✓ Recognition started (2 threads running)");
            return true;
            
        } catch (SecurityException e) {
            Log.e(TAG, "No microphone permission", e);
            if (callback != null) {
                callback.onError("No microphone permission");
            }
            return false;
        } catch (Exception e) {
            Log.e(TAG, "Failed to start recording", e);
            if (callback != null) {
                callback.onError("Failed to start recording: " + e.getMessage());
            }
            return false;
        }
    }
    
    /**
     * 停止识别
     */
    public void stopRecognition() {
        if (!isRecording) {
            Log.w(TAG, "Recognition not started");
            return;
        }
        
        Log.i(TAG, "Stopping recognition...");
        isRecording = false;
        
        // 1. 停止录音线程
        if (audioRecord != null) {
            try {
                audioRecord.stop();
                audioRecord.release();
                audioRecord = null;
            } catch (Exception e) {
                Log.e(TAG, "Error stopping AudioRecord", e);
            }
        }
        
        if (recordThread != null) {
            try {
                recordThread.join(1000);
            } catch (InterruptedException e) {
                Log.w(TAG, "Interrupted while waiting for record thread", e);
            }
            recordThread = null;
        }
        
        // 2. ✅ 等待推理完成（处理剩余音频）
        Log.i(TAG, "Waiting for inference to complete...");
        WhisperJNI.finishInput();
        WhisperJNI.waitForCompletion();
        
        // 3. ✅ 停止结果收集线程
        isCollectingResults = false;
        if (resultThread != null) {
            try {
                resultThread.join(2000);  // 等待最多2秒
            } catch (InterruptedException e) {
                Log.w(TAG, "Interrupted while waiting for result thread", e);
            }
            resultThread = null;
        }
        
        // 4. ✅ 打印统计信息
        int inferenceCount = WhisperJNI.getInferenceCount();
        double avgTime = WhisperJNI.getAverageInferenceTime();
        Log.i(TAG, String.format("✓ Recognition stopped (inferences=%d, avg_time=%.1fms)", 
                                 inferenceCount, avgTime));
        
        // 5. ✅ 通知完成
        if (callback != null) {
            callback.onComplete();
        }
    }
    
    /**
     * 释放资源
     */
    public void release() {
        Log.i(TAG, "Releasing Whisper resources...");
        
        // 先停止识别
        if (isRecording) {
            stopRecognition();
        }
        
        // 释放 JNI 资源
        WhisperJNI.releaseSession();
        
        callback = null;
        
        Log.i(TAG, "✓ Resources released");
    }
    
    /**
     * 获取完整识别结果
     */
    public String getFullResult() {
        return WhisperJNI.getAccumulatedResult();
    }
    
    /**
     * 清空累积的识别结果
     */
    public void clearResult() {
        WhisperJNI.resetAccumulatedResult();
    }
    
    /**
     * ✅ 获取推理统计信息
     */
    public String getStatistics() {
        int count = WhisperJNI.getInferenceCount();
        double avgTime = WhisperJNI.getAverageInferenceTime();
        return String.format("Inferences: %d, Avg time: %.1f ms", count, avgTime);
    }
    
    /**
     * 录音循环（在后台线程运行）
     * ✅ 修改：只负责喂数据，不再等待结果
     */
private void recordingLoop() {
    Log.i(TAG, "Recording thread started");
    
    short[] buffer = new short[BUFFER_SIZE_PER_READ];
    int feedCount = 0;
    StringBuilder accumulatedText = new StringBuilder();  // ✅ 添加本地累积
    
    while (isRecording) {
        try {
            int readSize = audioRecord.read(buffer, 0, buffer.length);
            
            if (readSize > 0) {
                feedCount++;
                
                // 调用 JNI 输入音频数据
                String result = WhisperJNI.feedAudio(buffer);
                
                // 如果有新识别结果，回调通知
                if (result != null && !result.isEmpty()) {
                    accumulatedText.append(result);  // ✅ 本地累积
                    Log.i(TAG, "New result: " + result);
                    Log.d(TAG, "Accumulated: " + accumulatedText.toString());
                    
                    if (callback != null) {
                        callback.onNewText(result);
                    }
                }
                
                if (feedCount % 50 == 0) {
                    Log.d(TAG, "Fed " + feedCount + " audio chunks");
                }
            } catch (Exception e) {
                Log.e(TAG, "[RecordThread] Error", e);
                if (callback != null) {
                    callback.onError("Recording error: " + e.getMessage());
                }
                break;
            }
        }
    }
    
    Log.i(TAG, "Recording thread finished");
    Log.i(TAG, "Total accumulated text: " + accumulatedText.toString());
}
    
    /**
     * ✅ 新增：结果收集循环（在独立线程运行）
     * 持续从推理线程获取结果并回调
     */
    private void resultCollectionLoop() {
        Log.i(TAG, "[ResultThread] Started");
        
        int resultCount = 0;
        
        while (isCollectingResults) {
            try {
                // ✅ 方案1：非阻塞轮询（适合高频率回调）
                String result = WhisperJNI.tryGetResult();
                
                if (result != null && !result.isEmpty()) {
                    resultCount++;
                    Log.i(TAG, "[ResultThread] New result #" + resultCount + ": " + result);
                    
                    if (callback != null) {
                        callback.onNewText(result);
                    }
                } else {
                    // 没有结果时短暂休眠，避免忙等
                    Thread.sleep(50);
                }
                
                // ✅ 方案2：阻塞等待（推荐，更省CPU）
                // 取消上面的代码，启用下面这段：
                /*
                String result = WhisperJNI.getResult(100);  // 100ms超时
                
                if (result != null && !result.isEmpty()) {
                    resultCount++;
                    Log.i(TAG, "[ResultThread] New result #" + resultCount + ": " + result);
                    
                    if (callback != null) {
                        callback.onNewText(result);
                    }
                }
                */
                
            } catch (InterruptedException e) {
                Log.w(TAG, "[ResultThread] Interrupted");
                break;
            } catch (Exception e) {
                Log.e(TAG, "[ResultThread] Error", e);
                if (callback != null) {
                    callback.onError("Result collection error: " + e.getMessage());
                }
                break;
            }
        }
        
        // ✅ 线程结束前，收集所有剩余结果
        Log.i(TAG, "[ResultThread] Collecting remaining results...");
        String result;
        int remainingCount = 0;
        while ((result = WhisperJNI.tryGetResult()) != null && !result.isEmpty()) {
            remainingCount++;
            Log.i(TAG, "[ResultThread] Remaining result #" + remainingCount + ": " + result);
            if (callback != null) {
                callback.onNewText(result);
            }
        }
        
        Log.i(TAG, String.format("[ResultThread] Finished (collected %d results, %d remaining)", 
                                resultCount, remainingCount));
    }
}