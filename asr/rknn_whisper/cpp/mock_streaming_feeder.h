#ifndef MOCK_STREAMING_FEEDER_H
#define MOCK_STREAMING_FEEDER_H

#include <vector>
#include <functional>

#include "audio_utils.h"  // read_audio / audio_buffer_t

// 模拟实时流式输入：
//  1. LoadFile() 读取音频文件，缓存为 float 样本。
//  2. StartFeeding() 按 interval_sec 的时间间隔回调一段样本。
//
// 注意：重构后移除了 timestamp 参数，因为新的 StreamingManager 
//       基于样本数而非时间戳触发推理。
class MockStreamingFeeder {
public:
    MockStreamingFeeder() = default;
    ~MockStreamingFeeder() = default;

    // 加载音频文件
    bool LoadFile(const char* path);

    // 开始喂送数据
    // - sample_rate: 音频采样率
    // - interval_sec: 每次回调的时间间隔（模拟实时输入）
    // - callback: 回调函数 (data, count)
    void StartFeeding(
        int sample_rate,
        float interval_sec,
        const std::function<void(const float*, size_t)>& callback) const;

private:
    std::vector<float> m_samples;
};

#endif // MOCK_STREAMING_FEEDER_H