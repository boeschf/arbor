#include <chrono>
#include <memory>
#include <functional>
#include <utility>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <arbor/export.hpp>

namespace arb {
namespace gpu {

/// Device queries

using DeviceProp = cudaDeviceProp;

struct ARB_SYMBOL_VISIBLE api_error_type {
    cudaError_t value;
    api_error_type(): value(cudaSuccess) {}
    api_error_type(cudaError_t e): value(e) {}

    operator bool() const {
        return value==cudaSuccess;
    }

    bool is_invalid_device() const {
        return value == cudaErrorInvalidDevice;
    }

    std::string name() const {
        std::string s = cudaGetErrorName(value);
        return s;
    }

    std::string description() const {
        std::string s = cudaGetErrorString(value);
        return s;
    }
};

constexpr auto gpuMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
constexpr auto gpuMemcpyHostToDevice = cudaMemcpyHostToDevice;
constexpr auto gpuMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
constexpr auto gpuHostRegisterPortable = cudaHostRegisterPortable;

template <typename... ARGS>
inline api_error_type get_device_properties(ARGS &&... args) {
    return cudaGetDeviceProperties(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type set_device(ARGS &&... args) {
    return cudaSetDevice(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_memcpy(ARGS &&... args) {
    return cudaMemcpy(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type host_register(ARGS &&... args) {
    return cudaHostRegister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type host_unregister(ARGS &&... args) {
    return cudaHostUnregister(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_malloc(ARGS &&... args) {
    return cudaMalloc(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_free(ARGS &&... args) {
    return cudaFree(std::forward<ARGS>(args)...);
}

template <typename... ARGS>
inline api_error_type device_mem_get_info(ARGS &&... args) {
    return cudaMemGetInfo(std::forward<ARGS>(args)...);
}


namespace detail {

struct callback_holder {
    std::function<void(api_error_type)> f;
    std::shared_ptr<callback_holder> self_ref;

    template<typename F>
    callback_holder(F&& func): f{std::forward<F>(func)} {}

    static void stream_callback(cudaStream_t /*stream*/, cudaError_t status, void* data) {
        // increase use_cout to 2
        std::shared_ptr<callback_holder> _this = reinterpret_cast<callback_holder*>(data)->self_ref;
        // decrease use_count to 1
        _this->self_ref.reset();
        _this->f(api_error_type{status});
        // clean up resources at exit
    }
};

} // namespace detail

//api_error_type add_callback(void(cb*)(void*), void* user_data) {
template<typename F>
api_error_type add_callback(F&& f) {
    auto h = std::make_shared<detail::callback_holder>(std::forward<F>(f));
    // add self-reference to avoid destruction at exit
    h->self_ref = h;
    return cudaStreamAddCallback(0, detail::callback_holder::stream_callback, h.get(), 0);
}

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
        else return (cudaEventQuery(event_) == cudaSuccess);
    }

    friend duration operator-(time_point& a, time_point& b);

private:
    time_point(cudaEvent_t e) : event_{e} {}

    void sync() {
        if (valid() && !synchronized_) {
            cudaEventSynchronize(event_);
            synchronized_ = true;
        }
    }

    bool valid() const {  return (bool)event_; }

    cudaEvent_t event_ = nullptr;
    bool synchronized_ = false;
};

time_point::duration operator-(time_point& end, time_point& start) {
    if (!(start.valid() && end.valid())) return time_point::duration(0.0f);
    start.sync();
    end.sync();
    float m;
    cudaEventElapsedTime(&m, start.event_, end.event_);
    return time_point::duration(m);
};

class clock {
public:
    using duration = time_point::duration;

    static time_point now() {
        cudaEvent_t event;
        cudaEventCreate(&event);
        cudaEventRecord(event);
        return {event};
    }
};

} // namespace chrono

#ifdef __CUDACC__
/// Atomics

// Wrappers around CUDA addition functions.
// CUDA 8 introduced support for atomicAdd with double precision, but only for
// Pascal GPUs (__CUDA_ARCH__ >= 600). These wrappers provide a portable
// atomic addition interface that chooses the appropriate implementation.

#if __CUDA_ARCH__ < 600 // Maxwell or older (no native double precision atomic addition)
__device__
inline double gpu_atomic_add(double* address, double val) {
    using I = unsigned long long int;
    I* address_as_ull = (I*)address;
    I old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val+__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else // use build in atomicAdd for double precision from Pascal onwards
__device__
inline double gpu_atomic_add(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

__device__
inline double gpu_atomic_sub(double* address, double val) {
    return gpu_atomic_add(address, -val);
}

__device__
inline float gpu_atomic_add(float* address, float val) {
    return atomicAdd(address, val);
}

__device__
inline float gpu_atomic_sub(float* address, float val) {
    return atomicAdd(address, -val);
}

/// Warp-Level Primitives

__device__ __inline__ double shfl(unsigned mask, double x, int lane)
{
    auto tmp = static_cast<uint64_t>(x);
    auto lo = static_cast<unsigned>(tmp);
    auto hi = static_cast<unsigned>(tmp >> 32);
    hi = __shfl_sync(mask, static_cast<int>(hi), lane, warpSize);
    lo = __shfl_sync(mask, static_cast<int>(lo), lane, warpSize);
    return static_cast<double>(static_cast<uint64_t>(hi) << 32 |
                               static_cast<uint64_t>(lo));
}

__device__ __inline__ unsigned ballot(unsigned mask, unsigned is_root) {
    return __ballot_sync(mask, is_root);
}

__device__ __inline__ unsigned any(unsigned mask, unsigned width) {
    return __any_sync(mask, width);
}

#ifdef __NVCC__
__device__ __inline__ double shfl_up(unsigned mask, int idx, unsigned lane_id, unsigned shift) {
    return __shfl_up_sync(mask, idx, shift);
}

__device__ __inline__ double shfl_down(unsigned mask, int idx, unsigned lane_id, unsigned shift) {
    return __shfl_down_sync(mask, idx, shift);
}

#else
__device__ __inline__ double shfl_up(unsigned mask, int idx, unsigned lane_id, unsigned shift) {
    return shfl(mask, idx, lane_id - shift);
}

__device__ __inline__ double shfl_down(unsigned mask, int idx, unsigned lane_id, unsigned shift) {
    return shfl(mask, idx, lane_id + shift);
}
#endif
#endif

} // namespace gpu
} // namespace arb
