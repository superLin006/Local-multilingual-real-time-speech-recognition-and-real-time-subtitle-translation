#include "mock_streaming_feeder.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <cstdlib>

bool MockStreamingFeeder::LoadFile(const char* path) {
    audio_buffer_t audio{};
    if (read_audio(path, &audio) != 0) {
        return false;
    }

    if (!audio.data || audio.num_frames <= 0 || audio.num_channels <= 0) {
        if (audio.data) {
            free(audio.data);
        }
        return false;
    }

    const size_t total_samples = static_cast<size_t>(audio.num_frames) * audio.num_channels;
    m_samples.assign(audio.data, audio.data + total_samples);

    free(audio.data);
    audio.data = nullptr;

    return !m_samples.empty();
}

void MockStreamingFeeder::StartFeeding(
    int sample_rate,
    float interval_sec,
    const std::function<void(const float*, size_t)>& callback) const {
    if (m_samples.empty() || !callback || sample_rate <= 0 || interval_sec <= 0.f) {
        return;
    }

    const size_t chunk_size = std::max<size_t>(
        1,
        static_cast<size_t>(std::llround(interval_sec * static_cast<float>(sample_rate)))
    );

    const auto start_time = std::chrono::steady_clock::now();
    double elapsed_sec = 0.0;

    size_t offset = 0;
    while (offset < m_samples.size()) {
        const size_t remaining = m_samples.size() - offset;
        const size_t count = std::min(chunk_size, remaining);

        auto ts = start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(elapsed_sec));
        callback(m_samples.data() + offset, count);

        offset += count;
        elapsed_sec += interval_sec;

        if (count == chunk_size) {
            std::this_thread::sleep_for(std::chrono::duration<double>(interval_sec));
        }
    }
}
