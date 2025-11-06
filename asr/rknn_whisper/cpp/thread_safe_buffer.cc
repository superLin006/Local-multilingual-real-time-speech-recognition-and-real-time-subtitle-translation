#include "thread_safe_buffer.h"
#include <algorithm>
#include <chrono>

ThreadSafeAudioBuffer::ThreadSafeAudioBuffer(size_t capacity)
    : m_capacity(capacity)
    , m_write_pos(0)
    , m_read_pos(0)
    , m_available(0)
    , m_finished(false) {
    m_buffer.resize(capacity, 0.0f);
}

bool ThreadSafeAudioBuffer::Push(const float* data, size_t count) {
    if (!data || count == 0) return false;
    
    std::lock_guard<std::mutex> lock(m_mutex);
    
    if (m_available + count > m_capacity) {
        size_t overflow = (m_available + count) - m_capacity;
        m_read_pos = (m_read_pos + overflow) % m_capacity;
        m_available = m_capacity - count;
    }
    
    size_t first_chunk = std::min(count, m_capacity - m_write_pos);
    std::memcpy(m_buffer.data() + m_write_pos, data, first_chunk * sizeof(float));
    
    if (count > first_chunk) {
        size_t second_chunk = count - first_chunk;
        std::memcpy(m_buffer.data(), data + first_chunk, second_chunk * sizeof(float));
    }
    
    m_write_pos = (m_write_pos + count) % m_capacity;
    m_available += count;
    
    m_cv_data_available.notify_one();
    return true;
}

bool ThreadSafeAudioBuffer::Pop(size_t sample_count, std::vector<float>& out, int timeout_ms) {
    std::unique_lock<std::mutex> lock(m_mutex);
    
    if (timeout_ms < 0) {
        m_cv_data_available.wait(lock, [this, sample_count] {
            return m_available >= sample_count || m_finished;
        });
    } else {
        bool success = m_cv_data_available.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            [this, sample_count] {
                return m_available >= sample_count || m_finished;
            }
        );
        
        if (!success) return false;
    }
    
    if (m_available < sample_count) {
        if (m_finished && m_available > 0) {
            sample_count = m_available;
        } else {
            return false;
        }
    }
    
    out.resize(sample_count);
    
    size_t first_chunk = std::min(sample_count, m_capacity - m_read_pos);
    std::memcpy(out.data(), m_buffer.data() + m_read_pos, first_chunk * sizeof(float));
    
    if (sample_count > first_chunk) {
        size_t second_chunk = sample_count - first_chunk;
        std::memcpy(out.data() + first_chunk, m_buffer.data(), second_chunk * sizeof(float));
    }
    
    m_read_pos = (m_read_pos + sample_count) % m_capacity;
    m_available -= sample_count;
    
    return true;
}

size_t ThreadSafeAudioBuffer::AvailableSize() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_available;
}

void ThreadSafeAudioBuffer::SetFinished() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_finished = true;
    m_cv_data_available.notify_all();
}

bool ThreadSafeAudioBuffer::IsFinished() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_finished;
}

void ThreadSafeAudioBuffer::Clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_write_pos = 0;
    m_read_pos = 0;
    m_available = 0;
    m_finished = false;
}