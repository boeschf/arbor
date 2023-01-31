#include "backends/gpu/multi_event_stream.hpp"

namespace arb {
namespace gpu {

// These wrappers are implemented in the multi_event_stream.cu file, which
// is spearately compiled by nvcc, to protect nvcc from having to parse C++17.
void make_ranges_w(
    arb_size_type num_dt,
    arb_size_type num_mech_indices,
    const arb_size_type* sizes,
    const time_type* ev_time,
    const cell_local_size_type* mech_index,
    const time_type* dt,
    arb_deliverable_event_range* ranges);

void compress_w(
    arb_size_type num_dt,
    arb_size_type num_mech_indices,
    const arb_deliverable_event_range* ranges,
    arb_deliverable_event_range* ranges_transposed,
    arb_size_type* num_ranges);

void multi_event_stream::make_ranges() {
    make_ranges_w(
        num_dt_,
        num_mech_indices_,
        device_sizes_.data(),
        device_ev_time_.data(),
        device_mech_indices_.data(),
        device_dt_.data(),
        device_ranges_.data());
}

void multi_event_stream::compress() {
    compress_w(
        num_dt_,
        num_mech_indices_,
        device_ranges_.data(),
        device_ranges_transposed_.data(),
        device_num_ranges_.data());
}

} // namespace gpu
} // namespace arb
