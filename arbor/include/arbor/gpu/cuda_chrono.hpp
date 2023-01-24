#include <chrono>

#include <arbor/arbexcept.hpp>

#include "gpu_api.hpp"

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
        if (valid()) {
            api_error_type status = cudaEventDestroy(event_);
            if (!status) throw arbor_exception("cuda exception: " + status.name() + ": " + status.description());
        }
        event_ = std::exchange(other.event_, nullptr);
        synchronized_ = std::exchange(other.synchronized_, false);
        return *this;
    }

    ~time_point() {
        if (valid()) cudaEventDestroy(event_);
    }

    //bool ready() {
    //    if (!valid() || synchronized_) {
    //        return true;
    //    }
    //    else {
    //        auto ret = cudaEventQuery(event_);
    //        if (ret == cudaSuccess) {
    //            synchronized_ = true;
    //            return true;
    //        }
    //        else if (ret == cudaErrorNotReady) {
    //            return false;
    //        }
    //        else {
    //            api_error_type status(ret);
    //            throw arbor_exception("cuda exception: " + status.name() + ": " + status.description());
    //        }
    //    }
    //}

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
    if (!(start.valid() && end.valid())) {
        throw arbor_exception("cuda exception: event not valid");
    }
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

