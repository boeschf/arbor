#pragma once
#include <cstring>
#include <array>

#include "backends/gpu/gpu_store_types.hpp"

namespace arb {
namespace gpu {

namespace {
constexpr std::size_t cbprng_batch_size = 4;
}
        
void generate_normal_random_values(
    std::size_t   width,
    std::size_t   num_variables,
    std::uint64_t seed, 
    std::uint64_t mech_id,
    std::uint64_t counter,
    arb_size_type** prng_indices,
    std::array<arb_value_type**, cbprng_batch_size> dst
);

inline void generate_normal_random_values(
    std::size_t   width,
    std::uint64_t seed, 
    std::uint64_t mech_id,
    std::uint64_t counter,
    memory::device_vector<arb_size_type*>& prng_indices,
    std::vector<memory::device_vector<arb_value_type*>>& dst
)
{
    generate_normal_random_values(
        width,
        dst[0].size(),
        seed,
        mech_id,
        counter,
        prng_indices.data(),
        {dst[0].data(), dst[1].data(), dst[2].data(), dst[3].data()}
    );
}

} // namespace gpu
} // namespace arb
