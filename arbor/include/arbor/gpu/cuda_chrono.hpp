#include <chrono>

#include <arbor/arbexcept.hpp>

#include "cuda_api.hpp"

namespace arb {
namespace gpu {
namespace chrono {

class clock;

class time_point {
public:
    friend class clock;

    using duration = std::chrono::duration<float, std::milli>;

    time_point() = default;

    time_point(const time_point&) = delete;

    time_point(time_point&& other)
    : event_{std::exchange(other.event_, nullptr)}
    , synchronized_{std::exchange(other.synchronized_, false)} {}

    time_point& operator=(const time_point&) = delete;

    time_point& operator=(time_point&& other) {
        if (valid()) cudaEventDestroy(event_);
        event_ = std::exchange(other.event_, nullptr);
        synchronized_ = std::exchange(other.synchronized_, false);
        return *this;
    }

    ~time_point() {
        if (valid()) cudaEventDestroy(event_);
    }

    bool ready() const {
        if (!valid() || synchronized_) return true;
        else {
            api_error_type status = cudaEventQuery(event_);
            if (status) return true;
            else throw arbor_exception("cuda exception: " + status.name() + ": " + status.description());
        }
    }

    friend duration operator-(time_point& a, time_point& b);

private:
    time_point(cudaEvent_t e) : event_{e} {}

    void sync() {
        if (valid() && !synchronized_) {
            api_error_type status = cudaEventSynchronize(event_);
            if (!status) throw arbor_exception("cuda exception: " + status.name() + ": " + status.description());
            synchronized_ = true;
        }
    }

    bool valid() const {  return (bool)event_; }

    cudaEvent_t event_ = nullptr;
    bool synchronized_ = false;
};

inline time_point::duration operator-(time_point& end, time_point& start) {
    if (!(start.valid() && end.valid())) return time_point::duration(0.0f);
    start.sync();
    end.sync();
    float m;
    api_error_type status = cudaEventElapsedTime(&m, start.event_, end.event_);
    if (!status) throw arbor_exception("cuda exception: " + status.name() + ": " + status.description());
    return time_point::duration(m);
};

class clock {
public:
    using duration = time_point::duration;

    static time_point now() {
        cudaEvent_t event;
        {
            api_error_type status = cudaEventCreate(&event);
            if (!status) throw arbor_exception("cuda exception: " + status.name() + ": " + status.description());
        }
        {
            api_error_type status = cudaEventRecord(event);
            if (!status) throw arbor_exception("cuda exception: " + status.name() + ": " + status.description());
        }
        return {event};
    }
};

} // namespace chrono
} // namespace gpu
} // namespace arb

