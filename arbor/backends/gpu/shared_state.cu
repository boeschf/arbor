// GPU kernels and wrappers for shared state methods.

#include <cstdint>

#include <backends/event.hpp>
#include <backends/event_stream_state.hpp>

#include <arbor/gpu/gpu_api.hpp>
#include <arbor/gpu/gpu_common.hpp>

namespace arb {
namespace gpu {

namespace kernel {

// Vector/scalar addition: x[i] += v ∀i
template <typename T>
__global__ void add_scalar(unsigned n,
                           T* __restrict__ const x,
                           arb_value_type v) {
    unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i<n) {
        x[i] += v;
    }
}

__global__ void take_samples_impl(
    const raw_probe_info* __restrict__ const begin_marked,
    const raw_probe_info* __restrict__ const end_marked,
    const arb_value_type time,
    arb_value_type* __restrict__ const sample_time,
    arb_value_type* __restrict__ const sample_value)
{
    const unsigned i = threadIdx.x+blockIdx.x*blockDim.x;
    const unsigned nsamples = end_marked - begin_marked;
    if (i<nsamples) {
        const auto p = begin_marked+i;
        sample_time[p->offset] = time;
        sample_value[p->offset] = p->handle? *p->handle: 0;
    }
}

} // namespace kernel

using impl::block_count;

void add_scalar(std::size_t n, arb_value_type* data, arb_value_type v) {
    if (!n) return;

    constexpr int block_dim = 128;
    const int nblock = block_count(n, block_dim);
    kernel::add_scalar<<<nblock, block_dim>>>(n, data, v);
}

void take_samples_impl(
    const event_stream_state<raw_probe_info>& s,
    const arb_value_type& time, arb_value_type* sample_time, arb_value_type* sample_value)
{
    constexpr int block_dim = 128;
    const int nsamples = s.size();
    if (nsamples) {
        const int nblock = block_count(nsamples, block_dim);
        kernel::take_samples_impl<<<nblock, block_dim>>>(s.begin_marked, s.end_marked, time, sample_time, sample_value);
    }
}

} // namespace gpu
} // namespace arb
