#include "result_queue.h"
#include <chrono>

void ResultQueue::Push(const std::string& result) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_queue.push(result);
    m_cv.notify_one();
}

bool ResultQueue::Pop(std::string& result, int timeout_ms) {
    std::unique_lock<std::mutex> lock(m_mutex);
    
    if (timeout_ms < 0) {
        m_cv.wait(lock, [this] { return !m_queue.empty(); });
    } else {
        bool success = m_cv.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            [this] { return !m_queue.empty(); }
        );
        if (!success) return false;
    }
    
    if (m_queue.empty()) return false;
    
    result = m_queue.front();
    m_queue.pop();
    return true;
}

bool ResultQueue::TryPop(std::string& result) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_queue.empty()) return false;
    
    result = m_queue.front();
    m_queue.pop();
    return true;
}

size_t ResultQueue::Size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.size();
}

void ResultQueue::Clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    while (!m_queue.empty()) {
        m_queue.pop();
    }
}