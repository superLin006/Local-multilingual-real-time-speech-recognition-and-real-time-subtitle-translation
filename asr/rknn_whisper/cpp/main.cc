#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <android/log.h>
#include <fstream>      // ← 添加这行
#include <algorithm>    // ← 添加这行 (for std::remove)

#include "whisper.h"
#include "audio_utils.h"
#include "process.h"
#include "streaming_manager.h"
#include "mock_streaming_feeder.h"

#define LOG_TAG    "WhisperDemo"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...)  __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

struct PerformanceStats {
    uint64_t total_inference_time_us;
    uint32_t inference_count;
    double npu_load_percent;
    size_t memory_weight_kb;
    size_t memory_internal_kb;
    size_t memory_total_kb;
    
    PerformanceStats() : total_inference_time_us(0), inference_count(0), 
                        npu_load_percent(0.0), memory_weight_kb(0),
                        memory_internal_kb(0), memory_total_kb(0) {}
};

double read_npu_load() {
    std::ifstream load_file("/sys/kernel/debug/rknpu/load");
    if (!load_file.is_open()) {
        load_file.open("/proc/debug/rknpu/load");
        if (!load_file.is_open()) {
            return -1.0;
        }
    }
    
    std::string line;
    if (std::getline(load_file, line)) {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            std::string percent_str = line.substr(pos + 1);
            
            try {
                percent_str.erase(std::remove(percent_str.begin(), percent_str.end(), ' '), percent_str.end());
                percent_str.erase(std::remove(percent_str.begin(), percent_str.end(), '%'), percent_str.end());
                
                if (percent_str.empty() || !std::isdigit(percent_str[0])) {
                    return -1.0;
                }
                
                return std::stod(percent_str);
            } catch (const std::exception& e) {
                return -1.0;
            }
        }
    }
    return -1.0;
}

bool query_model_memory(rknn_app_context_t* app_ctx, PerformanceStats& stats) {
    if (!app_ctx || !app_ctx->rknn_ctx) {
        return false;
    }
    
    rknn_mem_size mem_size;
    memset(&mem_size, 0, sizeof(mem_size));
    
    int ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_MEM_SIZE, &mem_size, sizeof(mem_size));
    if (ret != RKNN_SUCC) {
        LOGE("Failed to query memory size, ret=%d", ret);
        return false;
    }
    
    stats.memory_weight_kb = mem_size.total_weight_size / 1024;
    stats.memory_internal_kb = mem_size.total_internal_size / 1024;
    stats.memory_total_kb = (mem_size.total_weight_size + mem_size.total_internal_size) / 1024;
    
    return true;
}

void print_performance_summary(const PerformanceStats& stats, 
                              double total_process_time_sec,
                              float total_audio_duration_sec) {
    LOGI("\n");
    LOGI("╔══════════════════════════════════════════════════╗");
    LOGI("║            Performance & Memory Summary          ║");
    LOGI("╚══════════════════════════════════════════════════╝");
    
    LOGI("Memory Usage:");
    LOGI("  Weight Memory:       %zu KB (%.2f MB)", 
         stats.memory_weight_kb, stats.memory_weight_kb / 1024.0);
    LOGI("  Internal Memory:     %zu KB (%.2f MB)", 
         stats.memory_internal_kb, stats.memory_internal_kb / 1024.0);
    LOGI("  Total Memory:        %zu KB (%.2f MB)", 
         stats.memory_total_kb, stats.memory_total_kb / 1024.0);
    
    if (stats.npu_load_percent >= 0) {
        LOGI("NPU Load:              %.1f%%", stats.npu_load_percent);
    }
    
    LOGI("\nInference Performance:");
    LOGI("  Total inferences:    %u", stats.inference_count);
    
    if (stats.inference_count > 0) {
        double avg_time_ms = (stats.total_inference_time_us / 1000.0) / stats.inference_count;
        LOGI("  Avg inference time:  %.2f ms", avg_time_ms);
    }
    
    if (total_audio_duration_sec > 0.0f) {
        float rtf = total_process_time_sec / total_audio_duration_sec;
        LOGI("\nReal-Time Factor (RTF):");
        LOGI("  Audio duration:      %.2f seconds", total_audio_duration_sec);
        LOGI("  Process time:        %.2f seconds", total_process_time_sec);
        LOGI("  RTF:                 %.3f", rtf);
        
        if (rtf < 1.0f) {
            LOGI("  Performance:         ✓ %.2fx faster than realtime", 1.0f / rtf);
        } else {
            LOGI("  Performance:         ✗ %.2fx slower than realtime", rtf);
        }
    }
    
    LOGI("══════════════════════════════════════════════════\n");
}

static std::map<std::string,int> lang2code = {
    {"en",50259}, {"zh",50260}, {"de",50261}, {"es",50262}, {"ru",50263},
    {"ko",50264}, {"fr",50265}, {"ja",50266}, {"pt",50267}, {"tr",50268},
    {"pl",50269}, {"ca",50270}, {"nl",50271}, {"ar",50272}, {"sv",50273},
    {"it",50274}, {"id",50275}, {"hi",50276}, {"fi",50277}, {"vi",50278},
    {"he",50279}, {"uk",50280}, {"el",50281}, {"ms",50282}, {"cs",50283},
    {"ro",50284}, {"da",50285}, {"hu",50286}, {"ta",50287}, {"no",50288},
    {"th",50289}, {"ur",50290}, {"hr",50291}, {"bg",50292}, {"lt",50293},
    {"la",50294}, {"mi",50295}, {"ml",50296}, {"cy",50297}, {"sk",50298},
    {"te",50299}, {"fa",50300}, {"lv",50301}, {"bn",50302}, {"sr",50303},
    {"az",50304}, {"sl",50305}, {"kn",50306}, {"et",50307}, {"mk",50308},
    {"br",50309}, {"eu",50310}, {"is",50311}, {"hy",50312}, {"ne",50313},
    {"mn",50314}, {"bs",50315}, {"kk",50316}, {"sq",50317}, {"sw",50318},
    {"gl",50319}, {"mr",50320}, {"pa",50321}, {"si",50322}, {"km",50323},
    {"sn",50324}, {"yo",50325}, {"so",50326}, {"af",50327}, {"oc",50328},
    {"ka",50329}, {"be",50330}, {"tg",50331}, {"sd",50332}, {"gu",50333},
    {"am",50334}, {"yi",50335}, {"lo",50336}, {"uz",50337}, {"fo",50338},
    {"ht",50339}, {"ps",50340}, {"tk",50341}, {"nn",50342}, {"mt",50343},
    {"sa",50344}, {"lb",50345}, {"my",50346}, {"bo",50347}, {"tl",50348},
    {"mg",50349}, {"as",50350}, {"tt",50351}, {"haw",50352}, {"ln",50353},
    {"ha",50354}, {"ba",50355}, {"jw",50356}, {"su",50357},
    {"translate",50358}, {"transcribe",50359},
    {"startoflm",50360}, {"startofprev",50361},
    {"nospeech",50362}, {"notimestamps",50363}
};

static rknn_whisper_context_t* g_rknn_ctx = nullptr;
static StreamingManager* g_stream_manager = nullptr;
static float* g_mel_filters = nullptr;
static VocabEntry* g_vocab = nullptr;
static bool g_initialized = false;
static std::string g_accumulated_result;
static PerformanceStats g_perf_stats;
static std::string g_last_result;  // 用于 JNI 非阻塞获取结果

extern "C" int init_session(
    const char* encoder_path,
    const char* decoder_path, 
    const char* vocab_path,
    const char* mel_filters_path,
    const char* task
) {
    if (g_initialized) {
        LOGE("Session already initialized!");
        return -1;
    }
    
    LOGI("========================================");
    LOGI("Initializing Whisper Session (Async)...");
    LOGI("========================================");
    
    int ret = 0;
    int task_code = 0;
    StreamingConfig cfg;
    std::map<std::string,int>::iterator it;
    PerformanceStats encoder_stats;
    PerformanceStats decoder_stats;
    
    g_rknn_ctx = new rknn_whisper_context_t{};
    g_mel_filters = (float*)malloc(N_MELS * MELS_FILTERS_SIZE * sizeof(float));
    g_vocab = new VocabEntry[VOCAB_NUM]{};
    
    if (!g_mel_filters || !g_vocab) {
        LOGE("Memory allocation failed!");
        ret = -1;
        goto cleanup;
    }
    
    LOGI("Loading mel filters from: %s", mel_filters_path);
    ret = read_mel_filters(mel_filters_path, g_mel_filters, N_MELS * MELS_FILTERS_SIZE);
    if (ret != 0) {
        LOGE("Failed to load mel filters!");
        goto cleanup;
    }
    LOGI("✓ Mel filters loaded");
    
    LOGI("Loading vocabulary from: %s", vocab_path);
    ret = read_vocab(vocab_path, g_vocab);
    if (ret != 0) {
        LOGE("Failed to load vocabulary!");
        goto cleanup;
    }
    LOGI("✓ Vocabulary loaded");
    
    LOGI("Initializing encoder from: %s", encoder_path);
    ret = init_whisper_model(encoder_path, &g_rknn_ctx->encoder_context);
    if (ret != 0) {
        LOGE("Failed to init encoder!");
        goto cleanup;
    }
    LOGI("✓ Encoder initialized");
    
    LOGI("Initializing decoder from: %s", decoder_path);
    ret = init_whisper_model(decoder_path, &g_rknn_ctx->decoder_context);
    if (ret != 0) {
        LOGE("Failed to init decoder!");
        goto cleanup;
    }
    LOGI("✓ Decoder initialized");
    
    if (query_model_memory(&g_rknn_ctx->encoder_context, encoder_stats)) {
        g_perf_stats.memory_weight_kb = encoder_stats.memory_weight_kb;
        g_perf_stats.memory_internal_kb = encoder_stats.memory_internal_kb;
        g_perf_stats.memory_total_kb = encoder_stats.memory_total_kb;
    }
    
    if (query_model_memory(&g_rknn_ctx->decoder_context, decoder_stats)) {
        g_perf_stats.memory_weight_kb += decoder_stats.memory_weight_kb;
        g_perf_stats.memory_internal_kb += decoder_stats.memory_internal_kb;
        g_perf_stats.memory_total_kb += decoder_stats.memory_total_kb;
    }
    
    it = lang2code.find(task);
    if (it == lang2code.end()) {
        LOGE("Unsupported language: '%s'", task);
        ret = -1;
        goto cleanup;
    }
    task_code = it->second;
    
    cfg = StreamingConfig::FromTime(6.0f, 4.0f, SAMPLE_RATE);
    
    g_stream_manager = new StreamingManager(
        cfg, g_rknn_ctx, g_mel_filters, g_vocab, task_code
    );
    
    g_initialized = true;
    g_accumulated_result.clear();
    
    LOGI("========================================");
    LOGI("Session initialized successfully!");
    LOGI("========================================\n");
    return 0;
    
cleanup:
    if (g_mel_filters) { 
        free(g_mel_filters); 
        g_mel_filters = nullptr; 
    }
    
    if (g_vocab) { 
        for (int i = 0; i < VOCAB_NUM; i++) {
            if (g_vocab[i].token) {
                free(g_vocab[i].token);
            }
        }
        delete[] g_vocab; 
        g_vocab = nullptr; 
    }
    
    if (g_rknn_ctx) {
        release_whisper_model(&g_rknn_ctx->encoder_context);
        release_whisper_model(&g_rknn_ctx->decoder_context);
        delete g_rknn_ctx;
        g_rknn_ctx = nullptr;
    }
    
    return ret;
}

extern "C" int feed_audio(
    const float* samples,
    size_t count,
    char* result_buffer,
    size_t buffer_size
) {
    if (!g_initialized || !g_stream_manager) {
        LOGE("Session not initialized!");
        return -1;
    }
    
    // ✅ 如果传入了样本数据，则喂入
    if (samples && count > 0) {
        g_stream_manager->FeedAsync(samples, count);
    }
    
    // ✅ 尝试获取新结果（非阻塞）
    std::string result;
    if (g_stream_manager->TryGetResult(result)) {
        if (!result.empty()) {
            // ✅ 累积结果（这行很关键！）
            g_accumulated_result += result;
            
            if (result_buffer && buffer_size > 0) {
                strncpy(result_buffer, result.c_str(), buffer_size - 1);
                result_buffer[buffer_size - 1] = '\0';
            }
            
            return 1;  // 有新结果
        }
    }
    
    if (result_buffer && buffer_size > 0) {
        result_buffer[0] = '\0';
    }
    return 0;  // 无新结果
}

extern "C" int release_session() {
    if (!g_initialized) {
        return 0;
    }
    
    LOGI("========================================");
    LOGI("Releasing Session...");
    LOGI("========================================");
    
    if (g_stream_manager) {
        delete g_stream_manager;
        g_stream_manager = nullptr;
    }
    
    if (g_rknn_ctx) {
        release_whisper_model(&g_rknn_ctx->encoder_context);
        release_whisper_model(&g_rknn_ctx->decoder_context);
        delete g_rknn_ctx;
        g_rknn_ctx = nullptr;
    }
    
    if (g_mel_filters) {
        free(g_mel_filters);
        g_mel_filters = nullptr;
    }
    
    if (g_vocab) {
        for (int i = 0; i < VOCAB_NUM; i++) {
            if (g_vocab[i].token) {
                free(g_vocab[i].token);
            }
        }
        delete[] g_vocab;
        g_vocab = nullptr;
    }
    
    g_initialized = false;
    g_accumulated_result.clear();
    
    LOGI("Session released\n");
    return 0;
}

extern "C" const char* get_accumulated_result() {
    return g_accumulated_result.c_str();
}

extern "C" void reset_accumulated_result() {
    g_accumulated_result.clear();
}

// 阻塞获取结果（带超时）
extern "C" const char* get_result_blocking(int timeout_ms) {
    if (!g_initialized || !g_stream_manager) {
        return nullptr;
    }
    
    std::string result;
    if (g_stream_manager->GetResult(result, timeout_ms)) {
        if (!result.empty()) {
            g_accumulated_result += result;
            g_last_result = result;
            return g_last_result.c_str();
        }
    }
    
    return nullptr;
}

// 标记输入结束
extern "C" void finish_audio_input() {
    if (g_stream_manager) {
        g_stream_manager->FinishInput();
    }
}

// 等待推理完成
extern "C" void wait_for_inference_completion() {
    if (g_stream_manager) {
        g_stream_manager->WaitForCompletion();
    }
}

// 获取推理次数
extern "C" int get_inference_count() {
    if (g_stream_manager) {
        return g_stream_manager->GetInferenceCount();
    }
    return 0;
}

// 获取平均推理时间
extern "C" double get_average_inference_time() {
    if (g_stream_manager) {
        return g_stream_manager->GetAverageInferenceTime();
    }
    return 0.0;
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        LOGE("Usage: %s <encoder> <decoder> <task> <audio> <vocab> <mel_filters>", argv[0]);
        return -1;
    }

    const char *encoder_path     = argv[1];
    const char *decoder_path     = argv[2];
    const char *task             = argv[3];
    const char *audio_path       = argv[4];
    const char *vocab_path       = argv[5];
    const char *mel_filters_path = argv[6];

    LOGI("\n╔══════════════════════════════════════════════════╗");
    LOGI("║    Whisper Async Streaming Demo                 ║");
    LOGI("╚══════════════════════════════════════════════════╝");

    int ret = init_session(encoder_path, decoder_path, vocab_path, 
                          mel_filters_path, task);
    if (ret != 0) {
        return -1;
    }

    MockStreamingFeeder feeder;
    if (!feeder.LoadFile(audio_path)) {
        LOGE("Failed to load: %s", audio_path);
        release_session();
        return -1;
    }

    audio_buffer_t audio_for_duration{};
    float total_audio_duration_sec = 0.0f;
    if (read_audio(audio_path, &audio_for_duration) == 0) {
        total_audio_duration_sec = audio_for_duration.num_frames / (float)SAMPLE_RATE;
        if (audio_for_duration.data) free(audio_for_duration.data);
    }

    LOGI("Audio: %.2fs, Starting feed...\n", total_audio_duration_sec);
    
    auto stream_start = std::chrono::high_resolution_clock::now();
    
    // ✅ 启动结果收集线程
    std::atomic<bool> result_thread_running(true);
    std::thread result_thread([&]() {
        std::string result;
        int result_count = 0;
        
        while (result_thread_running.load()) {
            if (g_stream_manager->GetResult(result, 100)) {
                if (!result.empty()) {
                    result_count++;
                    
                    // ✅ 累积结果
                    g_accumulated_result += result;
                    
                    LOGI(">>> OUTPUT #%d: '%s'", result_count, result.c_str());
                    printf(">>> OUTPUT #%d: '%s'\n", result_count, result.c_str());
                }
            }
        }
        
        // ✅ 线程结束前，清空剩余结果
        LOGI("Result thread finishing, draining remaining results...");
        while (g_stream_manager->TryGetResult(result)) {
            if (!result.empty()) {
                result_count++;
                g_accumulated_result += result;
                LOGI(">>> OUTPUT #%d: '%s'", result_count, result.c_str());
                printf(">>> OUTPUT #%d: '%s'\n", result_count, result.c_str());
            }
        }
        
        LOGI("Result thread collected %d results", result_count);
    });
    
    // ✅ 喂入数据（同步阻塞）
    feeder.StartFeeding(SAMPLE_RATE, 0.1f, [&](const float* data, size_t count) {
        g_stream_manager->FeedAsync(data, count);
    });
    
    LOGI("Audio feeding completed, waiting for inference...");
    
    // ✅ 标记输入结束并等待推理完成
    g_stream_manager->FinishInput();
    g_stream_manager->WaitForCompletion();
    
    LOGI("Inference completed, stopping result thread...");
    
    // ✅ 停止结果线程
    result_thread_running = false;
    if (result_thread.joinable()) {
        result_thread.join();
    }
    
    auto stream_end = std::chrono::high_resolution_clock::now();
    double total_time_sec = std::chrono::duration<double>(stream_end - stream_start).count();

    g_perf_stats.inference_count = g_stream_manager->GetInferenceCount();
    g_perf_stats.npu_load_percent = read_npu_load();
    
    print_performance_summary(g_perf_stats, total_time_sec, total_audio_duration_sec);
    
    const char* full_result = get_accumulated_result();
    LOGI("\n╔══════════════════════════════════════════════════╗");
    LOGI("║            Final Transcription                   ║");
    LOGI("╚══════════════════════════════════════════════════╝");
    
    if (full_result && full_result[0] != '\0') {
        printf("%s\n", full_result);
        LOGI("%s", full_result);
    } else {
        printf("(empty)\n");
        LOGI("(empty result - no transcription generated)");
    }
    
    LOGI("══════════════════════════════════════════════════\n");

    release_session();
    return 0;
}