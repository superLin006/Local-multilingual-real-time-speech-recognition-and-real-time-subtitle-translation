#ifndef THREAD_SAFE_BUFFER_H
#define THREAD_SAFE_BUFFER_H

#include <vector>
#include <mutex>
#include <condition_variable>
#include <cstring>

class ThreadSafeAudioBuffer {
public:
    explicit ThreadSafeAudioBuffer(size_t capacity);
    
    // 非阻塞写入（生产者调用）
    bool Push(const float* data, size_t count);
    
    // 阻塞读取（消费者调用，等待直到有足够数据或超时）
    bool Pop(size_t sample_count, std::vector<float>& out, int timeout_ms = -1);
    
    // 获取当前可用数据量
    size_t AvailableSize() const;
    
    // 标记输入结束
    void SetFinished();
    bool IsFinished() const;
    
    // 清空缓冲区
    void Clear();
    
private:
    std::vector<float> m_buffer;
    size_t m_capacity;
    size_t m_write_pos;
    size_t m_read_pos;
    size_t m_available;
    bool m_finished;
    
    mutable std::mutex m_mutex;
    std::condition_variable m_cv_data_available;
};

#endif // THREAD_SAFE_BUFFER_H