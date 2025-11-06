#pragma once
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include "whisper.h"
#include "audio_utils.h"
#include "thread_safe_buffer.h"
#include "result_queue.h"

struct StreamingConfig {
    size_t chunk_size;
    size_t context_size;
    
    static StreamingConfig FromTime(float chunk_sec, float context_sec, int sample_rate) {
        StreamingConfig cfg;
        cfg.chunk_size = static_cast<size_t>(chunk_sec * sample_rate);
        cfg.context_size = static_cast<size_t>(context_sec * sample_rate);
        return cfg;
    }
    
    size_t GetHopSize() const {
        return chunk_size > context_size ? (chunk_size - context_size) : chunk_size;
    }
};

class SlidingWindowBuffer {
public:
    explicit SlidingWindowBuffer(size_t capacity);
    void Push(const float* data, size_t count);
    size_t Size() const { return m_size; }
    size_t Capacity() const { return m_capacity; }
    bool Extract(size_t sample_count, std::vector<float>& out) const;
    void KeepTail(size_t keep_count);
    void Clear();
    
private:
    std::vector<float> m_storage;
    size_t m_capacity;
    size_t m_size;
};

class StreamingManager {
public:
    StreamingManager(const StreamingConfig& cfg,
                     rknn_whisper_context_t* ctx,
                     float* mel_filters,
                     VocabEntry* vocab,
                     int task_code);
    
    ~StreamingManager();
    
    // 非阻塞输入音频（生产者接口）
    void FeedAsync(const float* samples, size_t count);
    
    // 尝试获取推理结果（非阻塞）
    bool TryGetResult(std::string& result);
    
    // 阻塞获取推理结果（带超时）
    bool GetResult(std::string& result, int timeout_ms = -1);
    
    // 标记输入结束（用于离线文件处理）
    void FinishInput();
    
    // 等待所有推理完成
    void WaitForCompletion();
    
    // 统计信息
    size_t GetBufferSize() const { return m_audio_buffer.AvailableSize(); }
    int GetInferenceCount() const { return m_inference_count.load(); }
    double GetAverageInferenceTime() const;
    size_t GetResultQueueSize() const { return m_result_queue.Size(); }
    
    void ResetState();
    
private:
    void InferenceLoop();
    std::string RunInference(const std::vector<float>& chunk);
    std::string ExtractNewText(const std::string& full_text);
    std::string TrimWhitespace(const std::string& str) const;
    
    StreamingConfig m_cfg;
    rknn_whisper_context_t* m_ctx;
    float* m_mel_filters;
    VocabEntry* m_vocab;
    int m_task_code;
    
    ThreadSafeAudioBuffer m_audio_buffer;
    ResultQueue m_result_queue;
    
    std::thread m_inference_thread;
    std::atomic<bool> m_running;
    
    SlidingWindowBuffer m_context_buffer;
    size_t m_accumulated_new_samples;
    size_t m_min_trigger_samples;
    
    std::string m_last_full_text;
    whisper_perf_stats_t m_perf_stats;
    
    std::atomic<int> m_inference_count;
    std::atomic<uint64_t> m_total_inference_time_us;
};