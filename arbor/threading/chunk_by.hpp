#pragma once

#include <arbor/assert.hpp>

#include <threading/threading.hpp>
#include <threading/scan.hpp>
#include <threading/enumerable_thread_specific.hpp>

#include <cstdint>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <vector>
#include <optional>
#include <ranges>
#include <functional>
#include <iostream>
#include <syncstream>


namespace arb {
namespace threading {
namespace detail {

struct interval {
    bool flag         = true; // true if binary predicate on input range is true
    std::size_t index = 0u;   // plus scan of inverse flags
    std::size_t count = 0u;   // counts elements with this index
    std::size_t end   = 0u;   // counts all elements
};

// plus operator for scan
inline interval operator+(interval l, interval r) noexcept {
    return {
        r.flag,
        l.index + r.index,
        r.index ? r.count : l.count + r.count,
        l.end + r.end};
}

// example (serial)
//
// range indices            -  0  1  2  3  4  5  6  7  8  9  -
// input range:                1  1  2  2  2  2  3  4  4  5
//
// make intervals: invoke predicate for each pair, add special first and last intervals
// index is inverse of flag, count and end are 1
//   flag:                  1  1  0  1  1  1  0  0  1  0  0
//   index:                 0  0  1  0  0  0  1  1  0  1  1
//   count:                 1  1  1  1  1  1  1  1  1  1  1
//   end:                   1  1  1  1  1  1  1  1  1  1  1
//
// scan intervals (inclusive) with plus operator
//   flag:                  1  1  0  1  1  1  0  0  1  0  0
//   index:                 0  0  1  1  1  1  2  3  3  4  5
//   count:                 1  2  1  2  3  4  1  1  2  1  1
//   end:                   1  2  3  4  5  6  7  8  9 10 11
//
// for each interval pair: output range [l.end-l.count, l.end) at index l.index if !r.flag
// !r.flag                     0  1  0  0  0  1  1  0  1  1
// l.index                        0           1  2     3  4
// l.end - l.count                0           2  6     7  9
// l.end                          2           6  7     9 10
//
// out[0] = range(0,  2) -> {1, 1}
// out[1] = range(2,  6) -> {2, 2, 2, 2}
// out[2] = range(6,  7) -> {3}
// out[3] = range(7,  9) -> {4, 4}
// out[4] = range(9, 10) -> {5}

} // namespace detail {

template<typename InputRange>
using chunk_by_output_range = std::vector<std::remove_cvref_t<
    decltype(stdr::subrange(stdr::begin(std::declval<InputRange&>()), stdr::end(std::declval<InputRange&>())))>>;

template<typename InputRange>
auto make_chunk_vector(InputRange&& in) {
    using In = std::remove_cvref_t<InputRange>;
    using sub_range = std::remove_cvref_t<decltype(
        stdr::subrange(stdr::begin(std::declval<In&>()), stdr::end(std::declval<In&>())))>;
    return std::vector<sub_range>{stdr::size(in)};
}

template<typename InputRange, typename BinaryOp>
auto chunk_by(execution_policy const& p, InputRange&& in, chunk_by_output_range<InputRange>& out, BinaryOp op) {
    // allocate enough space for output ranges
    out.resize(stdr::size(in));

    // return early for degenerate cases
    if (stdr::size(in) == 0u)
        return stdr::subrange(stdr::begin(out), stdr::end(out));
    if (stdr::size(in) == 1u) {
        out[0] = stdr::subrange(stdr::begin(in), stdr::end(in));
        return stdr::subrange(stdr::begin(out), stdr::end(out));
    }

    // interval types
    using interval = detail::interval;
    using interval_op_t = std::plus<interval>;
    static constexpr interval_op_t interval_op;

    // get the tiled input range
    auto range_for_tile = p.tile(in);
    std::size_t num_tiles = range_for_tile.num_tiles();

    // current tile
    std::atomic<std::size_t> tile_counter = 0u;
    // scan states for tiles (shared among threads)
    detail::scan_state<interval, interval_op_t> tile_states(num_tiles, interval_op);

    // thread local storage: local vector of intervals, bool for indicating sequential execution
    struct tl_scratch {
        std::vector<interval> intervals;
        bool sequential = true;
    };
    enumerable_thread_specific<tl_scratch> scratch(p.get_ts());

    // schedule tasks for each tile
    task_group g(&p.get_ts());
    for (std::size_t i=0u; i<num_tiles; ++i) {
        g.run([&]() {
            // look up the thread local items
            auto& [intervals, seq] = scratch.local();
            // get tile index
            auto tile = tile_counter.fetch_add(1u, std::memory_order_relaxed);
            // check if this is the last tile
            bool is_last_tile = tile == num_tiles-1;
            // get the tile's corresponding subrange from the input range
            auto sub_range = range_for_tile[tile];
            // include the subrange's left neighbor
            if (tile > 0) sub_range = stdr::subrange{--stdr::begin(sub_range), stdr::end(sub_range)};
            // allocate enough space for the intervals
            intervals.resize(stdr::size(sub_range) + is_last_tile);

            // loop over pairs of input range items and make intervals
            // TODO for c++23: this should really be stdr::transform(sub_range | stdr::views::adjacent<2>, stdr::begin(intervals), lambda)
            {
                auto left_range  = stdr::subrange{stdr::begin(sub_range), --stdr::end(sub_range)};
                auto right_range = stdr::subrange{++stdr::begin(sub_range), stdr::end(sub_range)};
                auto i = ++stdr::begin(intervals); // leave out first interval
                for (auto left = stdr::begin(left_range); auto const& r : right_range) {
                    auto const& l = *left++;
                    bool b = std::invoke(op, l, r);
                    *i++ = interval{b, !b, 1, 1};
                }
            }
            // add last interval if this is the last tile
            if (is_last_tile) intervals.back() = interval{false,1u,1u,1u};

            // check whether this thread is in sequential or parallel mode
            if (auto seq_prefix = tile_states.is_sequential(tile, seq, interval{true, 0u, 1u, 1u}); seq_prefix) {
                // scan over the intervals and publish the last interval for other tiles to see
                intervals[0] = *seq_prefix;
                tile_states.set_global_prefix(tile,
                    *--std::inclusive_scan(intervals.begin(), intervals.end(), intervals.begin(), interval_op));
            }
            else {
                // reduce tile-local portion and publish local value
                tile_states.set_local_prefix(tile,
                    std::accumulate(intervals.begin()+2, intervals.end(), *(intervals.begin()+1), interval_op));
                // wait for predecessor prefix (decoupled look-back) and do local scan
                intervals[0] = tile_states.wait_for_predecessor_prefix(tile);
                std::inclusive_scan(intervals.begin(), intervals.end(), intervals.begin(), interval_op);

            }
            // loop over pairs of intervals and write to output
            // TODO for c++23: this should really be stdr::for_each(intervals | stdr::views::adjacent<2>, lambda)
            {
                auto left_range  = stdr::subrange{stdr::begin(intervals), --stdr::end(intervals)};
                auto right_range = stdr::subrange{++stdr::begin(intervals), stdr::end(intervals)};
                for (auto left = stdr::begin(left_range); auto const& r : right_range) {
                    auto const& l = *left++;
                    if (!r.flag) out[l.index] = stdr::subrange(stdr::next(stdr::begin(in), l.end - l.count),
                                                               stdr::next(stdr::begin(in), l.end));
                }
            }
        });
    }
    g.wait();

    //out.resize(stdr::size(in));
    // return a range of ranges
    return stdr::subrange(stdr::begin(out), stdr::next(stdr::begin(out), tile_states.back()->index));
}

} // namespace threading
} // namespace arb
