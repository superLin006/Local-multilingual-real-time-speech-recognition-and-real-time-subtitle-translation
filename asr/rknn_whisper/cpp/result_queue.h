#ifndef RESULT_QUEUE_H
#define RESULT_QUEUE_H

#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>

class ResultQueue {
public:
    ResultQueue() = default;
    
    void Push(const std::string& result);
    bool Pop(std::string& result, int timeout_ms = -1);
    bool TryPop(std::string& result);
    size_t Size() const;
    void Clear();
    
private:
    std::queue<std::string> m_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
};

#endif // RESULT_QUEUE_H