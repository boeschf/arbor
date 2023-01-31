#include <arbor/common_types.hpp>
#include <arbor/gpu/gpu_common.hpp>
#include <arbor/mechanism_abi.h>

namespace arb {
namespace gpu {

namespace kernels {

__global__
void make_ranges(
    const arb_size_type num_dt,
    const arb_size_type num_mech_indices,
    const arb_size_type* __restrict__ const sizes,
    const time_type* __restrict__ const ev_time,
    const cell_local_size_type* __restrict__ const mech_index,
    const time_type* __restrict__ const dt,
    arb_deliverable_event_range* __restrict__ const ranges) {

    const unsigned tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid < num_mech_indices) {
        const auto m = mech_index[tid];
        auto i = sizes[tid];
        const auto end = sizes[tid+1];
        for (arb_size_type j = 0; j < num_dt; ++j) {
            const auto t = dt[j+1];
            auto* r = ranges + (tid*num_dt+j);
            *r = arb_deliverable_event_range{m, i, i};
            while(i < end) {
                if (ev_time[i] < t) {
                    ++(r->end);
                    ++i;
                }
                else {
                    break;
                }
            }
        }
    }
}

__global__
void compress(
    const arb_size_type num_dt,
    const arb_size_type num_mech_indices,
    const arb_deliverable_event_range* __restrict__ const ranges,
    arb_deliverable_event_range* __restrict__ const ranges_transposed,
    arb_size_type* __restrict__ const num_ranges) {

    const unsigned tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid < num_dt) {
        arb_size_type count = 0;
        auto* r = ranges_transposed + tid*num_mech_indices;
        for (arb_size_type i = 0; i < num_mech_indices; ++i) {
            auto* r_next = ranges + tid + i*num_dt;
            if ((r_next->end - r_next->begin)) {
                *r = *r_next;
                ++r;
                ++count;
            }
        }
        num_ranges[tid] = count;
    }
}

} // namespace kernels

void make_ranges_w(
    arb_size_type num_dt,
    arb_size_type num_mech_indices,
    const arb_size_type* sizes,
    const time_type* ev_time,
    const cell_local_size_type* mech_index,
    const time_type* dt,
    arb_deliverable_event_range* ranges) {
    const int nblock = impl::block_count(num_mech_indices, 128);
    kernels::make_ranges<<<nblock, 128>>>(num_dt, num_mech_indices, sizes, ev_time, mech_index, dt, ranges);
}

void compress_w(
    arb_size_type num_dt,
    arb_size_type num_mech_indices,
    const arb_deliverable_event_range* ranges,
    arb_deliverable_event_range* ranges_transposed,
    arb_size_type* num_ranges) {
    const int nblock = impl::block_count(num_dt, 128);
    kernels::compress<<<nblock, 128>>>(num_dt, num_mech_indices, ranges, ranges_transposed, num_ranges);
}

} // namespace gpu
} // namespace arb
