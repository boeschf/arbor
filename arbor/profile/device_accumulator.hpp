#pragma once

#include <array>
#include <cassert>
#include <chrono>
#include <functional>
#include <future>
#include <memory>

#include <arbor/arbexcept.hpp>
#include <arbor/gpu/chrono.hpp>
#include <arbor/profile/profiler.hpp>

namespace arb {
namespace profile {

namespace detail {

template<std::size_t N>
struct device_accumulator {
    using size_type = unsigned;
    using timer_type = timer<>;

    struct record {
        tick_type host_start_time;
        tick_type host_end_time;
        ::arb::gpu::chrono::time_point start_time;
        ::arb::gpu::chrono::time_point end_time;
        //std::promise<tick_type> start_promise, end_promise;
        //std::future<tick_type> start_future, end_future;

        void reset() {
            host_start_time = timer_type::tic();
            start_time = ::arb::gpu::chrono::clock::now();
            end_time = ::arb::gpu::chrono::time_point{};
            //start_promise = std::promise<tick_type>{};
            //end_promise = std::promise<tick_type>{};
            //start_future = start_promise.get_future();
            //end_future = end_promise.get_future();
        }
    };

    static size_type increment(size_type i) noexcept { return (i + 1) % N; }
    static size_type decrement(size_type i) noexcept { return ((i + N) - 1) % N; }
    size_type start() const noexcept { return (current + capacity) % N; }
    size_type size() const noexcept { return N - capacity; }

    // start timing
    void tic() {
        if (recording) throw arbor_exception("device_accumulator: toc() has not been called");
        recording = true;
        // wait for one timing to finish if there is no space
        if (capacity == 0) accumulate(records[current]);
        // prepare the current record and invoke the launch function
        auto& r = records[current];
        r.reset();
        current = increment(current);
        --capacity;
        //auto tic_callback = [p=&r](::arb::gpu::api_error_type status) {
        //    if (status) {
        //        p->start_promise.set_value(timer_type::tic());
        //    }
        //    else {
        //        p->start_promise.set_exception(
        //            std::make_exception_ptr(
        //                arbor_exception("device_accumulator: " + status.description())));
        //    }
        //};
        //if (auto status = ::arb::gpu::add_callback(tic_callback); !status) {
        //    throw arbor_exception("device_accumulator: " + status.description());
        //}
    }

    // end timing
    void toc() {
        if (!recording) throw arbor_exception("device_accumulator: tic() has not been called");
        recording = false;
        auto& r = records[decrement(current)];
        r.host_end_time = timer_type::tic();
        r.end_time = ::arb::gpu::chrono::clock::now();
        //auto toc_callback = [p=&r](::arb::gpu::api_error_type status) {
        //    if (status) {
        //        p->end_promise.set_value(timer_type::tic());
        //    }
        //    else {
        //        p->start_promise.set_exception(
        //            std::make_exception_ptr(
        //                arbor_exception("device_accumulator: " + status.description())));
        //    }
        //};
        //if (auto status = ::arb::gpu::add_callback(toc_callback); !status) {
        //    throw arbor_exception("device_accumulator: " + status.description());
        //}
        //check();
    }

    void accumulate(record& r) {
        const double elapsed_device = (r.end_time - r.start_time).count() * 1.0e-3; 
        //const auto start_time = r.start_future.get();
        //const auto end_time = r.end_future.get();
        ////const double elapsed_host = (end_time-r.host_start_time)*timer_type::seconds_per_tick();
        //const double elapsed_device = (end_time-start_time)*timer_type::seconds_per_tick();
        sum += elapsed_device;
        ++capacity;
    }

    //// check if any futures are ready and free up capacity
    //void check() {
    //    using namespace std::chrono_literals;
    //    const auto end = current;
    //    for(auto begin = start(); begin != end; increment(begin)) {
    //        auto& r = records[begin];
    //        //if (r.start_time.ready() && r.end_time.ready()) {
    //        if (r.end_time.ready()) {
    //        //if(auto status = r.end_future.wait_for(10us); status == std::future_status::ready) {
    //            accumulate(r);
    //        }
    //        else {
    //            break;
    //        }
    //    }
    //}

    // wait for all currently pending timings to finish
    void wait() {
        if (recording) throw arbor_exception("device_accumulator: toc() has not been called");
        current = start();
        const size_type n = size();
        for (size_type i = 0; i < n; ++i) {
            accumulate(records[current]);
            current = increment(current);
        }
        assert(capacity == N);
    }

    // return accumulated value by waiting for all currently pending timings
    double get() {
        wait();
        return sum;
    }

    // wait for all pending timings to finsish and reset the sum to zero
    void reset() {
        wait();
        sum = 0.;
    }

    std::array<record,N> records;
    size_type current = 0u;
    size_type capacity = N;
    double sum = 0.;
    bool recording = false;
};

} // namespace detail

template<std::size_t N = 16>
struct device_accumulator {
public:
    device_accumulator() : m{std::make_unique<detail::device_accumulator<N>>()} {}

    device_accumulator(device_accumulator&&) = default;

    device_accumulator& operator=(device_accumulator&&) = default;

    void tic() { m->tic(); }

    void toc() { m->toc(); }

    double get() const {
        return const_cast<detail::device_accumulator<N>*>(m.get())->get(); }
        //return m->get(); }

    void reset() { m->reset(); }

private:
    std::unique_ptr<detail::device_accumulator<N>> m;
};

} // namespace profile
} // namespace arb
