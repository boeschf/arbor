#pragma once

#include <arbor/assert.hpp>

#include <threading/threading.hpp>
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
#include <variant>


namespace arb {
namespace threading {

namespace stdr = std::ranges;

// execution_policy stores a pointer to the task system
// and a hint of how to divied the work into tiles.
// The hints can be specified in one of three ways
// 1. as number of tiles
// 2. as tile size (number of elements)
// 3. as tile size in bytes
//
// execution_policy has no public constructor, but offers
// static factory functions to create the object with a hint:
// - create_from_num_tiles(ts, n): fix number of tiles
// - create_from_tile_size(ts, n): fix tile size
// - create_from_tile_size_bytes(ts, n): fix tile size (in bytes)
// - create(ts): default tiling
//
// The class has two public methods:
// - get_ts(): returns a reference to the task system
// - tile(r): get a tiled_range object for range r
//
// The class template tiled_range has two public methods:
// - num_tiles(): returns the number of tiles
// - operator[](i): returns the ith subrange of range r
struct execution_policy {
    static execution_policy create_from_num_tiles(task_system& ts, std::size_t n) noexcept {
        return {ts, from_t{std::in_place_type<from_num_tiles_>, n}}; }
    static execution_policy create_from_tile_size(task_system& ts, std::size_t n) noexcept {
        return {ts, from_t{std::in_place_type<from_tile_size_>, n}}; }
    static execution_policy create_from_tile_size_bytes(task_system& ts, std::size_t n) noexcept {
        return {ts, from_t{std::in_place_type<from_tile_size_bytes_>, n}}; }
    static execution_policy create(task_system& ts) noexcept {
        return {ts, from_t{std::in_place_type<from_tile_size_bytes_>, std::hardware_destructive_interference_size}}; }

    task_system& get_ts() const noexcept { return *ts_ptr_; }

    template<typename Range>
    auto tile(Range& r) const noexcept { return tiled_range<std::remove_cvref_t<Range>>{r, from_}; }

  private:
    struct from_num_tiles_ { std::size_t v; };
    struct from_tile_size_ { std::size_t v; };
    struct from_tile_size_bytes_ { std::size_t v; };
    using from_t = std::variant<from_num_tiles_, from_tile_size_, from_tile_size_bytes_>;

    template<typename Range>
    struct tiled_range {
        tiled_range() = delete;
        tiled_range(tiled_range const&) = default;
        tiled_range(tiled_range &&) = default;

        auto operator[](std::size_t i) const noexcept {
            auto start = (i < num_tiles_hi_) ? (i*tile_size_hi_) :
                (num_tiles_hi_*tile_size_hi_ + (i-num_tiles_hi_)*tile_size_lo_);
            auto end   = (i < num_tiles_hi_) ? ((i+1u)*tile_size_hi_) :
                (num_tiles_hi_*tile_size_hi_ + (i-num_tiles_hi_+1u)*tile_size_lo_);
            return stdr::subrange(
                stdr::next(stdr::begin(r_), start),
                stdr::next(stdr::begin(r_), end));
        }

        auto num_tiles() const noexcept { return num_tiles_; }

      private:
        friend class execution_policy;

        void init_from_num_tiles(std::size_t n) {
            num_tiles_    = n;
            tile_size_lo_ = range_size_/num_tiles_;
            num_tiles_hi_ = range_size_ - num_tiles_*tile_size_lo_;
            tile_size_hi_ = num_tiles_hi_ ? tile_size_lo_ + 1 : tile_size_lo_;
        }

        template<class... Ts>
        struct overloaded : Ts... { using Ts::operator()...; };

        tiled_range(Range& r, from_t const & from) noexcept : r_{r}, range_size_{stdr::size(r)} {
            std::visit(overloaded{
                [this](from_num_tiles_ const& x) noexcept {
                    init_from_num_tiles(std::max(std::min(x.v, range_size_), std::size_t{1u}));},
                [this](from_tile_size_ const& x) noexcept {
                    auto tile_size = std::max(std::min(x.v, range_size_), std::size_t{1u});
                    init_from_num_tiles(std::max((range_size_ + tile_size - 1u)/tile_size, std::size_t{1u}));},
                [this](from_tile_size_bytes_ const& x) noexcept {
                    auto tile_size = std::max(std::min(x.v/sizeof(stdr::range_value_t<Range>), range_size_), std::size_t{1u});
                    init_from_num_tiles(std::max((range_size_ + tile_size - 1u)/tile_size, std::size_t{1u}));}}, from);
        }

        Range& r_;
        std::size_t range_size_   = 0u;
        std::size_t num_tiles_    = 0u;
        std::size_t num_tiles_hi_ = 0u;
        std::size_t tile_size_hi_ = 0u;
        std::size_t tile_size_lo_ = 0u;
    };

    execution_policy(task_system& ts, from_t&& f) noexcept : ts_ptr_{&ts}, from_{std::move(f)} {}

    task_system* ts_ptr_ = nullptr;
    from_t from_;
};


namespace detail {

// This template class holds the local and global prefix sums for the decoupled
// look-back scan algorithm. It also has member functions to publish current
// and wait for predecessor values. Synchronization is achieved by atomically
// updating a flag using appropriate memory ordering constraints.
template<typename T, typename BinaryOp>
struct scan_state {
    // A prefix for any tile i can be either unavailable (not computed yet),
    // locally available (local prefix sum) or globally available (global
    // prefix sum)
    enum status_flag : unsigned char { unavailable, local_available, global_available };

    // local and global prefix values do not exist at the beginning and are
    // represented as optionals. This eliminates the need for default
    // (identity) values.
    struct tile_prefix {
        std::optional<T> lval;
        std::optional<T> gval;
        std::atomic<status_flag> status = unavailable;
    };

    // vector of prefixes for all tiles
    std::vector<tile_prefix> prefixes;
    // the operation which is used for scanning
    BinaryOp op;

    // construct from number of tiles and binary operation
    scan_state(std::size_t num_tiles, BinaryOp op_) : prefixes(num_tiles), op{op_} {}

    // access to the last global prefix
    auto& back() { return prefixes.back().gval; }

    // publish local prefix
    // only called from parallel algorithm
    void set_local_prefix(std::size_t i, T&& v) {
        arb_assert(i > 0);
        auto& prefix = prefixes[i];
        prefix.lval = std::move(v);
        prefix.status.store(local_available, std::memory_order_release);
        prefix.status.notify_all();
    }

    // publish global prefix by looking back
    // only called from parallel algorithm
    // assumes local prefix is set
    T wait_for_predecessor_prefix(std::size_t i) {
        arb_assert(i > 0);
        // optional for result
        std::optional<T> pred;
        // decoupled look-back: walk the predecessors
        for (std::size_t p = i; p>0; --p) {
            auto& prefix = prefixes[p-1];
            // wait for prefix to become available
            prefix.status.wait(unavailable, std::memory_order_acquire);
            auto status = prefix.status.load(std::memory_order_acquire);
            // apply binary op with predecessor prefix as left left operand
            if (status == local_available) {
                // TODO for c++23: use monadic operations
                if (pred) pred = std::invoke(op, *prefix.lval, *pred);
                else pred = *prefix.lval;
            }
            else {
                // TODO for c++23: use monadic operations
                if (pred) pred = std::invoke(op, *prefix.gval, *pred);
                else pred = *prefix.gval;
                // stop if we find a global prefix
                break;
            }
        }
        // get ith prefix: local value is set (precondition), and publish
        // global value by adding global predecessor prefix obtained during
        // look-back. Note that i can never be zero by construction of the scan
        // algorithm (i==0 is alwas sequential), and therefore, the pred
        // optional is always engaged.
        auto& prefix = prefixes[i];
        arb_assert(prefix.lval.has_value());
        arb_assert(pred.has_value());
        prefix.gval = std::invoke(op, *pred, *prefix.lval);
        prefix.status.store(global_available, std::memory_order_release);
        prefix.status.notify_all();
        return *pred;
    }

    // publish global prefix
    // only called from sequential algorithm
    void set_global_prefix(std::size_t i, T v) {
        auto& prefix = prefixes[i];
        prefix.gval = std::move(v);
        prefix.status.store(global_available, std::memory_order_release);
        prefix.status.notify_all();
    }

    // determines whether this thread is executing the sequential algorithm
    // returns an optional global prefix and updates the bool-like parameter seq
    template<typename B, typename U>
    std::optional<T> is_sequential(std::size_t i, B& seq, U&& init) {
        // only sequential if previously sequental and the global
        // predecessor prefix is available
        seq = seq && (i==0 ||
            (prefixes[i-1].status.load(std::memory_order_acquire) == global_available));
        if (seq) return i==0 ? std::forward<U>(init) : *prefixes[i-1].gval;
        else return {};
    }
};

} // namespace detail

template<typename Range, typename BinaryOp, typename U>
auto inclusive_scan(execution_policy const& p, Range&& range, BinaryOp op, U&& init) {
    // return early for degenerate cases
    if (stdr::size(range) < 2u) return;

    // get the tiled range
    auto range_for_tile = p.tile(range);
    std::size_t num_tiles = range_for_tile.num_tiles();

    // current tile
    std::atomic<std::size_t> tile_counter = 0u;
    // scan states for tiles (shared among threads)
    detail::scan_state<stdr::range_value_t<std::remove_reference_t<Range>>, BinaryOp> tile_states(num_tiles, op);

    // thread local storage: bool for indicating sequential execution
    // note: enumerable_thread_specifc internally stores the values in a
    // std::vector. This is not great for bools, since they are infamously
    // stored as bit fields: instead of a reference we get a proxy object, and
    // different threads access the same byte. Thus, we use unsigned char
    // instead of bool here.
    enumerable_thread_specific<unsigned char> sequential(1, p.get_ts());

    // schedule tasks for each tile
    task_group g(&p.get_ts());
    for (std::size_t i=0u; i<num_tiles; ++i) {
        g.run([&]() {
            // get tile index
            auto tile = tile_counter.fetch_add(1u, std::memory_order_relaxed);
            // get the tile's corresponding subrange from the input range
            auto sub_range = range_for_tile[tile];
            //if (stdr::size(sub_range) < 1u) return;
            // check whether this thread is in sequential or parallel mode
            if (auto seq_prefix = tile_states.is_sequential(tile, sequential.local(), std::forward<U>(init)); seq_prefix) {
                // scan over the range and publish the last element for other tiles to see
                tile_states.set_global_prefix(tile,
                    *--std::inclusive_scan(sub_range.begin(), sub_range.end(), sub_range.begin(), op, std::move(*seq_prefix)));
            }
            else {
                // reduce tile-local portion and publish local value
                tile_states.set_local_prefix(tile, std::accumulate(sub_range.begin()+1, sub_range.end(), sub_range.front(), op));
                // wait for predecessor prefix (decoupled look-back) and do local scan
                std::inclusive_scan(sub_range.begin(), sub_range.end(), sub_range.begin(), op,
                    tile_states.wait_for_predecessor_prefix(tile));
            }
        });
    }
    g.wait();
}

template<typename Range, typename BinaryOp>
auto inclusive_scan(execution_policy const& p, Range&& range, BinaryOp op) {
    if (stdr::size(range) < 2u) return;
    return inclusive_scan(p, stdr::subrange(stdr::next(stdr::begin(range), 1u), stdr::end(range)),
        op, range.front());
}

} // namespace threading
} // namespace arb
