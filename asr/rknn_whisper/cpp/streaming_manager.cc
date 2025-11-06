#include "streaming_manager.h"
#include "process.h"
#include <algorithm>
#include <cstring>
#include <cctype>
#include <android/log.h>

#define LOG_TAG    "StreamingManager"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ==================== SlidingWindowBuffer 实现 ====================

SlidingWindowBuffer::SlidingWindowBuffer(size_t capacity)
    : m_capacity(capacity), m_size(0) {
    m_storage.resize(capacity, 0.0f);
}

void SlidingWindowBuffer::Push(const float* data, size_t count) {
    if (count == 0) return;
    
    if (count >= m_capacity) {
        std::memcpy(m_storage.data(), data + (count - m_capacity), m_capacity * sizeof(float));
        m_size = m_capacity;
    } else {
        if (m_size + count > m_capacity) {
            size_t shift_amount = m_size + count - m_capacity;
            std::memmove(m_storage.data(), 
                        m_storage.data() + shift_amount,
                        (m_size - shift_amount) * sizeof(float));
            m_size -= shift_amount;
        }
        std::memcpy(m_storage.data() + m_size, data, count * sizeof(float));
        m_size += count;
    }
}

bool SlidingWindowBuffer::Extract(size_t sample_count, std::vector<float>& out) const {
    if (sample_count > m_size) {
        return false;
    }
    out.assign(m_storage.begin() + (m_size - sample_count),
               m_storage.begin() + m_size);
    return true;
}

void SlidingWindowBuffer::KeepTail(size_t keep_count) {
    if (keep_count >= m_size) {
        return;
    }
    if (keep_count == 0) {
        m_size = 0;
        return;
    }
    std::memmove(m_storage.data(),
                m_storage.data() + (m_size - keep_count),
                keep_count * sizeof(float));
    m_size = keep_count;
}

void SlidingWindowBuffer::Clear() {
    m_size = 0;
}

// ==================== StreamingManager 实现 ====================

StreamingManager::StreamingManager(const StreamingConfig& cfg,
                                   rknn_whisper_context_t* ctx,
                                   float* mel_filters,
                                   VocabEntry* vocab,
                                   int task_code)
    : m_cfg(cfg)
    , m_ctx(ctx)
    , m_mel_filters(mel_filters)
    , m_vocab(vocab)
    , m_task_code(task_code)
    , m_audio_buffer(cfg.chunk_size * 3)
    , m_context_buffer(cfg.chunk_size * 2)
    , m_accumulated_new_samples(0)
    , m_min_trigger_samples(static_cast<size_t>(3.5f * SAMPLE_RATE))
    , m_inference_count(0)
    , m_total_inference_time_us(0)
    , m_running(true)
    , m_perf_stats{} {
    
    LOGI("StreamingManager initialized (Async Mode):");
    LOGI("  chunk_size: %zu samples (%.2fs)", 
         cfg.chunk_size, cfg.chunk_size / (float)SAMPLE_RATE);
    LOGI("  context_size: %zu samples (%.2fs)", 
         cfg.context_size, cfg.context_size / (float)SAMPLE_RATE);
    LOGI("  buffer_capacity: %zu samples", cfg.chunk_size * 3);
    
    m_inference_thread = std::thread(&StreamingManager::InferenceLoop, this);
}

StreamingManager::~StreamingManager() {
    m_running = false;
    m_audio_buffer.SetFinished();
    
    if (m_inference_thread.joinable()) {
        m_inference_thread.join();
    }
    
    LOGI("StreamingManager destroyed");
}

void StreamingManager::FeedAsync(const float* samples, size_t count) {
    if (!samples || count == 0) return;
    
    m_audio_buffer.Push(samples, count);
}

bool StreamingManager::TryGetResult(std::string& result) {
    return m_result_queue.TryPop(result);
}

bool StreamingManager::GetResult(std::string& result, int timeout_ms) {
    return m_result_queue.Pop(result, timeout_ms);
}

void StreamingManager::FinishInput() {
    m_audio_buffer.SetFinished();
}

void StreamingManager::WaitForCompletion() {
    FinishInput();
    
    if (m_inference_thread.joinable()) {
        m_inference_thread.join();
    }
}

void StreamingManager::ResetState() {
    m_audio_buffer.Clear();
    m_result_queue.Clear();
    m_context_buffer.Clear();
    m_accumulated_new_samples = 0;
    m_last_full_text.clear();
    m_inference_count = 0;
    m_total_inference_time_us = 0;
}

double StreamingManager::GetAverageInferenceTime() const {
    int count = m_inference_count.load();
    if (count == 0) return 0.0;
    return (m_total_inference_time_us.load() / 1000.0) / count;
}

void StreamingManager::InferenceLoop() {
    LOGI("[InferenceThread] Started");
    
    int local_inference_count = 0;
    int idle_loops = 0;  // ✅ 添加空闲计数器
    
    while (m_running.load()) {
        size_t buffer_size = m_audio_buffer.AvailableSize();
        size_t context_size = m_context_buffer.Size();
        size_t total_available = buffer_size + context_size;
        
        bool should_infer = false;
        size_t infer_chunk_size = 0;
        
        if (local_inference_count == 0) {
            // 首次推理：累积1.5秒
            if (total_available >= m_min_trigger_samples) {
                should_infer = true;
                infer_chunk_size = std::min(total_available, m_cfg.chunk_size);
                LOGI("[Strategy] ColdStart: %.2fs (buffer=%zu, context=%zu)", 
                     infer_chunk_size / (float)SAMPLE_RATE, buffer_size, context_size);
            }
        } else if (local_inference_count == 1) {
            // 第二次推理：快速建立窗口
            if (buffer_size >= m_min_trigger_samples) {
                should_infer = true;
                infer_chunk_size = std::min(total_available, m_cfg.chunk_size);
                LOGI("[Strategy] WarmUp: %.2fs (buffer=%zu, context=%zu)", 
                     infer_chunk_size / (float)SAMPLE_RATE, buffer_size, context_size);
            }
        } else {
            // 稳定推理：1.5秒hop
            if (buffer_size >= m_cfg.GetHopSize() && total_available >= m_cfg.chunk_size) {
                should_infer = true;
                infer_chunk_size = m_cfg.chunk_size;
                LOGI("[Strategy] Stable: %.2fs (buffer=%zu, context=%zu)", 
                     infer_chunk_size / (float)SAMPLE_RATE, buffer_size, context_size);
            }
        }
        
        if (should_infer) {
            idle_loops = 0;  // ✅ 重置空闲计数
            
            std::vector<float> chunk;
            
            // 从context取数据
            size_t from_context = std::min(context_size, infer_chunk_size);
            size_t from_buffer = infer_chunk_size - from_context;
            
            if (from_context > 0) {
                std::vector<float> context_part;
                m_context_buffer.Extract(from_context, context_part);
                chunk.insert(chunk.end(), context_part.begin(), context_part.end());
            }
            
            // 从buffer取数据
            if (from_buffer > 0) {
                std::vector<float> buffer_part;
                if (m_audio_buffer.Pop(from_buffer, buffer_part, 1000)) {  // ✅ 增加超时到1秒
                    chunk.insert(chunk.end(), buffer_part.begin(), buffer_part.end());
                } else {
                    LOGW("[InferenceThread] Failed to pop %zu samples", from_buffer);
                    continue;
                }
            }
            
            LOGI("========== Inference #%d ==========", local_inference_count + 1);
            auto inference_start = std::chrono::high_resolution_clock::now();
            
            std::string full_text = RunInference(chunk);
            
            auto inference_end = std::chrono::high_resolution_clock::now();
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                inference_end - inference_start).count();
            
            local_inference_count++;
            m_inference_count++;
            m_total_inference_time_us += duration_us;
            
            LOGI("[Time] %.0f ms (avg: %.0f ms)", 
                 duration_us / 1000.0, GetAverageInferenceTime());
            
            if (!full_text.empty()) {
                LOGI("[Raw] '%s'", full_text.c_str());
                std::string new_part = ExtractNewText(full_text);
                if (!new_part.empty()) {
                    m_result_queue.Push(new_part);
                    LOGI("[New] '%s'", new_part.c_str());
                } else {
                    LOGI("[New] (empty - duplicate or no change)");
                }
            } else {
                LOGI("[Raw] (empty result)");
            }
            
            // 智能保留策略
            size_t keep_size;
            if (local_inference_count <= 2) {
                keep_size = chunk.size();
            } else {
                keep_size = m_cfg.context_size;
            }
            
            m_context_buffer.Clear();
            if (keep_size > 0 && keep_size <= chunk.size()) {
                m_context_buffer.Push(chunk.data() + chunk.size() - keep_size, keep_size);
            }
            
            LOGI("===================================\n");
            
        } else {
            // ✅ 未触发推理：检查退出条件
            if (m_audio_buffer.IsFinished()) {
                if (buffer_size == 0 && context_size == 0) {
                    LOGI("[InferenceThread] Input finished and no data left, exiting");
                    break;
                }
                
                // ✅ 输入结束但还有数据：强制最后一次推理
                if (buffer_size > 0 || context_size > 0) {
                    LOGI("[InferenceThread] Input finished, forcing final inference with %zu samples", 
                         total_available);
                    
                    std::vector<float> final_chunk;
                    
                    // 取出所有剩余数据
                    if (context_size > 0) {
                        std::vector<float> context_part;
                        m_context_buffer.Extract(context_size, context_part);
                        final_chunk.insert(final_chunk.end(), context_part.begin(), context_part.end());
                    }
                    
                    if (buffer_size > 0) {
                        std::vector<float> buffer_part;
                        if (m_audio_buffer.Pop(buffer_size, buffer_part, 1000)) {
                            final_chunk.insert(final_chunk.end(), buffer_part.begin(), buffer_part.end());
                        }
                    }
                    
                    if (!final_chunk.empty()) {
                        LOGI("========== Final Inference ==========");
                        auto start = std::chrono::high_resolution_clock::now();
                        
                        std::string full_text = RunInference(final_chunk);
                        
                        auto end = std::chrono::high_resolution_clock::now();
                        auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            end - start).count();
                        
                        local_inference_count++;
                        m_inference_count++;
                        m_total_inference_time_us += duration_us;
                        
                        if (!full_text.empty()) {
                            std::string new_part = ExtractNewText(full_text);
                            if (!new_part.empty()) {
                                m_result_queue.Push(new_part);
                            }
                        }
                        
                        LOGI("=====================================\n");
                    }
                    
                    break;
                }
            }
            
            // ✅ 等待新数据（避免忙等）
            idle_loops++;
            if (idle_loops % 100 == 0) {
                LOGI("[InferenceThread] Waiting for data... (buffer=%zu, context=%zu, need=%zu)",
                     buffer_size, context_size, m_min_trigger_samples);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));  // ✅ 改为sleep而非Pop
        }
    }
    
    LOGI("[InferenceThread] Stopped (total inferences: %d)", local_inference_count);
}

std::string StreamingManager::RunInference(const std::vector<float>& chunk) {
    if (chunk.empty()) return {};
    
    audio_buffer_t audio;
    audio.data = const_cast<float*>(chunk.data());
    audio.num_frames = static_cast<int>(chunk.size());
    audio.num_channels = 1;
    audio.sample_rate = SAMPLE_RATE;
    
    LOGI("[Inference] Processing %.2fs audio (%d frames)", 
         chunk.size() / (float)SAMPLE_RATE, audio.num_frames);
    
    std::vector<float> mel(N_MELS * ENCODER_INPUT_SIZE, 0.0f);
    audio_preprocess(&audio, m_mel_filters, mel);
    
    std::vector<std::string> recognized;
    int ret = inference_whisper_model(m_ctx, mel, m_mel_filters, m_vocab, 
                                     m_task_code, recognized, &m_perf_stats);
    
    if (ret != 0) {
        LOGE("[Inference] Failed with error code: %d", ret);
        return {};
    }
    
    if (recognized.empty()) {
        LOGW("[Inference] Model returned empty result");
        return {};
    }
    
    return recognized.front();
}

std::string StreamingManager::ExtractNewText(const std::string& full_text) {
    std::string trimmed_full = TrimWhitespace(full_text);
    std::string trimmed_last = TrimWhitespace(m_last_full_text);
    
    if (trimmed_full == trimmed_last) {
        LOGI("[Diff] Identical text");
        return {};
    }
    
    if (trimmed_last.empty()) {
        m_last_full_text = full_text;
        LOGI("[Diff] First output");
        return full_text;
    }
    
    if (trimmed_full.size() > trimmed_last.size()) {
        if (trimmed_full.compare(0, trimmed_last.size(), trimmed_last) == 0) {
            std::string new_part = trimmed_full.substr(trimmed_last.size());
            m_last_full_text = full_text;
            LOGI("[Diff] Normal append: +%zu chars", new_part.size());
            return TrimWhitespace(new_part);
        }
    }
    
    size_t common_suffix_len = 0;
    size_t max_check = std::min(trimmed_full.size(), trimmed_last.size());
    
    for (size_t i = 1; i <= max_check; ++i) {
        if (trimmed_full[trimmed_full.size() - i] == 
            trimmed_last[trimmed_last.size() - i]) {
            common_suffix_len = i;
        } else {
            break;
        }
    }
    
    LOGI("[Diff] Common suffix: %zu chars", common_suffix_len);
    
    if (trimmed_full.size() > trimmed_last.size()) {
        size_t overlap_threshold = std::min(size_t(15), trimmed_last.size() / 2);
        
        for (size_t start = 0; start < trimmed_full.size() - overlap_threshold; ++start) {
            size_t match_len = 0;
            size_t max_match = std::min(trimmed_last.size(), trimmed_full.size() - start);
            
            for (size_t i = 0; i < max_match; ++i) {
                if (std::tolower(trimmed_full[start + i]) == 
                    std::tolower(trimmed_last[i])) {
                    match_len++;
                } else {
                    break;
                }
            }
            
            if (match_len >= trimmed_last.size() * 0.5 && match_len >= overlap_threshold) {
                LOGI("[Diff] Found prefix match at pos %zu, len %zu", start, match_len);
                
                std::string new_part;
                if (start > 0) {
                    new_part = trimmed_full.substr(0, start);
                }
                
                size_t after_match = start + match_len;
                if (after_match < trimmed_full.size()) {
                    if (!new_part.empty()) new_part += " ";
                    new_part += trimmed_full.substr(after_match);
                }
                
                if (!new_part.empty()) {
                    m_last_full_text = full_text;
                    return TrimWhitespace(new_part);
                } else {
                    m_last_full_text = full_text;
                    return {};
                }
            }
        }
    }
    
    if (common_suffix_len > 0) {
        size_t new_start_pos = trimmed_full.size() - common_suffix_len;
        
        if (new_start_pos == 0) {
            m_last_full_text = full_text;
            return {};
        }
        
        std::string new_part = trimmed_full.substr(0, new_start_pos);
        
        if (trimmed_last.find(new_part) != std::string::npos) {
            m_last_full_text = full_text;
            return {};
        }
        
        m_last_full_text = full_text;
        return TrimWhitespace(new_part);
    }
    
    size_t overlap_pos = std::string::npos;
    size_t min_overlap = std::min(size_t(10), trimmed_last.size() / 2);
    
    for (size_t len = trimmed_last.size(); len >= min_overlap; --len) {
        std::string suffix = trimmed_last.substr(trimmed_last.size() - len);
        overlap_pos = trimmed_full.find(suffix);
        if (overlap_pos != std::string::npos) {
            LOGI("[Diff] Found overlap at pos %zu, len %zu", overlap_pos, len);
            
            size_t new_start = overlap_pos + len;
            if (new_start < trimmed_full.size()) {
                std::string new_part = trimmed_full.substr(new_start);
                m_last_full_text = full_text;
                return TrimWhitespace(new_part);
            } else {
                m_last_full_text = full_text;
                return {};
            }
        }
    }
    
    m_last_full_text = full_text;
    return full_text;
}

std::string StreamingManager::TrimWhitespace(const std::string& str) const {
    if (str.empty()) return str;
    
    size_t start = 0;
    while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start]))) {
        ++start;
    }
    
    if (start == str.size()) return {};
    
    size_t end = str.size();
    while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1]))) {
        --end;
    }
    
    return str.substr(start, end - start);
}