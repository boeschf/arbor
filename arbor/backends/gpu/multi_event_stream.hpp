#pragma once

#include <vector>

#include <arbor/arbexcept.hpp>
#include <arbor/fvm_types.hpp>
#include <arbor/generic_event.hpp>
#include <arbor/mechanism_abi.h>

#include "event_map.hpp"
#include "memory/memory.hpp"
#include "timestep_range.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

namespace arb {
namespace gpu {

class multi_event_stream {
public:
    using size_type = arb_size_type;
    using event_type = ::arb::deliverable_event;

    using event_time_type = ::arb::event_time_type<event_type>;
    using event_data_type = ::arb::event_data_type<event_type>;
    using event_index_type = ::arb::event_index_type<event_type>;

    using host_size_array = std::vector<size_type>;
    using host_data_array = std::vector<event_data_type>;
    using host_time_array = std::vector<event_time_type>;
    using host_dt_array = std::vector<time_type>;
    using host_mech_array = std::vector<cell_local_size_type>;

    using device_size_array = memory::device_vector<size_type>;
    using device_ranges_array = memory::device_vector<arb_deliverable_event_range>;
    using device_data_array = memory::device_vector<event_data_type>;
    using device_time_array = memory::device_vector<event_time_type>;
    using device_dt_array = memory::device_vector<time_type>;
    using device_mech_array = memory::device_vector<cell_local_size_type>;

    multi_event_stream() = default;

    bool empty() const {
        return (index_ == 0 || index_ > num_dt_) || !num_ranges_[index_-1];
    }

    void clear() {
        num_dt_ = 0u;
        num_mech_indices_ = 0u;
        index_ = 0u;

        ev_data_.clear();
        ev_time_.clear();
        dt_.clear();
        sizes_.clear();
        num_ranges_.clear();
        mech_indices_.clear();
    }

    void init(const mechanism_event_map& staged, const timestep_range& dts) {
        using ::arb::event_time;

        clear();

        if (dts.empty()) return;
        num_dt_ = dts.size();

        if (staged.empty()) return;
        num_mech_indices_ = staged.size();

        // scan over events
        sizes_.reserve(staged.size()+1);
        sizes_.push_back(0u);
        for (auto& [mech_index, vec] : staged) {
            sizes_.push_back(sizes_.back() + vec.size());
        }
        // total number of events
        const size_type n = sizes_.back();
        if (!n) return;

        // move sizes and dts to device and allocate storage
        copy(sizes_, device_sizes_);
        resize(device_ranges_, num_mech_indices_*num_dt_);
        resize(device_ranges_transposed_, num_mech_indices_*num_dt_);
        resize(device_num_ranges_, num_dt_);
        mech_indices_.reserve(num_mech_indices_);
        num_ranges_.resize(num_dt_);
        ev_data_.reserve(n);
        ev_time_.reserve(n);
        dt_.reserve(num_dt_+1);
        dt_.push_back(dts[0].t0());
        for (const auto dt : dts) {
            dt_.push_back(dt.t1());
        }
        copy(dt_, device_dt_);

        // assign mech indices, event time and event data
        for (auto& [mech_index, vec] : staged) {
            mech_indices_.push_back(mech_index);
            for (const auto& ev : vec) {
                ev_time_.push_back(event_time(ev));
                ev_data_.push_back(event_data(ev));
            }
        }
        // move event time and event data to device
        copy(ev_time_, device_ev_time_);
        copy(mech_indices_, device_mech_indices_);

        // create ranges from events for each mechanism index
        make_ranges();
        // transpose and compress the ranges
        compress();

        // copy event data to device
        copy(ev_data_, device_ev_data_);
        // copy numbers of compressed ranges to host
        memory::copy(device_num_ranges_, num_ranges_);
    }

    void mark() {
        index_ += (index_ <= num_dt_ ? 1 : 0);
    }

    arb_deliverable_event_stream marked_events() const {
        if (empty()) return {0, nullptr, nullptr};
        return {
            num_ranges_[index_-1],
            device_ev_data_.data(),
            device_ranges_transposed_.data() + num_mech_indices_*(index_-1)
        };
    }

private:
    template<typename H, typename D>
    static void copy(const H& h, D& d) {
        resize(d, h.size());
        memory::copy_async(h, memory::make_view(d)(0u, h.size()));
    }

    template<typename D>
    static void resize(D& d, size_type s) {
        if (d.size() < s) {
            d = D(s);
        }
    }

    void make_ranges();
    void compress();

protected:
    size_type num_dt_ = 0u;
    size_type num_mech_indices_ = 0u;
    size_type index_ = 0u;

    host_data_array ev_data_;
    host_time_array ev_time_;
    host_dt_array dt_;
    host_size_array sizes_;
    host_size_array num_ranges_;
    host_mech_array mech_indices_;

    device_data_array device_ev_data_;
    device_time_array device_ev_time_;
    device_dt_array device_dt_;
    device_size_array device_sizes_;
    device_ranges_array device_ranges_;
    device_ranges_array device_ranges_transposed_;
    device_size_array device_num_ranges_;
    device_mech_array device_mech_indices_;
};

} // namespace gpu
} // namespace arb
