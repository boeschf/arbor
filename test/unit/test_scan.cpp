#include <threading/scan.hpp>
#include <threading/chunk_by.hpp>

#include <gtest/gtest.h>
#include "common.hpp"

//#include <iostream>
//#include <ostream>

#include <vector>
#include <algorithm>
#include <ranges>

TEST(scan, execution_policy_from_num_tiles) {
    arb::threading::task_system ts(1);

    auto check = [&ts](auto&& expected, std::optional<std::size_t> n = {}, int bound = 10) {
        auto r = std::ranges::iota_view{0, bound};
        auto num_tiles = n.value_or(expected.size());
        auto p = arb::threading::execution_policy::create_from_num_tiles(ts, num_tiles);
        auto t = p.tile(r);
        EXPECT_EQ(t.num_tiles(), expected.size());
        for (std::size_t i=0; auto& r : expected) {
            EXPECT_TRUE(std::ranges::equal(r, t[i++]));
        }
    };
    // degenerate case: zero tiles -> returns 1 tile for whole range
    check(std::vector{std::ranges::iota_view{0, 10}}, {0});
    // degenerate case: empty range -> returns 1 tile for empty range
    check(std::vector{std::ranges::iota_view{0, 0}}, {}, 0);
    // degenerate case: zero tiles, empty range -> returns 1 tile for empty range
    check(std::vector{std::ranges::iota_view{0, 0}}, {0}, 0);
    // degenerate case: more tiles than elements
    check(std::vector{std::ranges::iota_view{0, 1}}, {2}, 1);
    // 1 tile
    check(std::vector{std::ranges::iota_view{0, 10}});
    // 2 tiles
    check(std::vector{std::ranges::iota_view{0,  5},
                      std::ranges::iota_view{5, 10}});
    // 3 tiles
    check(std::vector{std::ranges::iota_view{0,  4},
                      std::ranges::iota_view{4,  7},
                      std::ranges::iota_view{7, 10}});
    // 4 tiles
    check(std::vector{std::ranges::iota_view{0,  3},
                      std::ranges::iota_view{3,  6},
                      std::ranges::iota_view{6,  8},
                      std::ranges::iota_view{8, 10}});
    // 5 tiles
    check(std::vector{std::ranges::iota_view{0,  2},
                      std::ranges::iota_view{2,  4},
                      std::ranges::iota_view{4,  6},
                      std::ranges::iota_view{6,  8},
                      std::ranges::iota_view{8, 10}});
    // 6 tiles
    check(std::vector{std::ranges::iota_view{0,  2},
                      std::ranges::iota_view{2,  4},
                      std::ranges::iota_view{4,  6},
                      std::ranges::iota_view{6,  8},
                      std::ranges::iota_view{8,  9},
                      std::ranges::iota_view{9, 10}});
    // 7 tiles
    check(std::vector{std::ranges::iota_view{0,  2},
                      std::ranges::iota_view{2,  4},
                      std::ranges::iota_view{4,  6},
                      std::ranges::iota_view{6,  7},
                      std::ranges::iota_view{7,  8},
                      std::ranges::iota_view{8,  9},
                      std::ranges::iota_view{9, 10}});
    // 8 tiles
    check(std::vector{std::ranges::iota_view{0,  2},
                      std::ranges::iota_view{2,  4},
                      std::ranges::iota_view{4,  5},
                      std::ranges::iota_view{5,  6},
                      std::ranges::iota_view{6,  7},
                      std::ranges::iota_view{7,  8},
                      std::ranges::iota_view{8,  9},
                      std::ranges::iota_view{9, 10}});
    // 9 tiles
    check(std::vector{std::ranges::iota_view{0,  2},
                      std::ranges::iota_view{2,  3},
                      std::ranges::iota_view{3,  4},
                      std::ranges::iota_view{4,  5},
                      std::ranges::iota_view{5,  6},
                      std::ranges::iota_view{6,  7},
                      std::ranges::iota_view{7,  8},
                      std::ranges::iota_view{8,  9},
                      std::ranges::iota_view{9, 10}});
    // 10 tiles
    check(std::vector{std::ranges::iota_view{0,  1},
                      std::ranges::iota_view{1,  2},
                      std::ranges::iota_view{2,  3},
                      std::ranges::iota_view{3,  4},
                      std::ranges::iota_view{4,  5},
                      std::ranges::iota_view{5,  6},
                      std::ranges::iota_view{6,  7},
                      std::ranges::iota_view{7,  8},
                      std::ranges::iota_view{8,  9},
                      std::ranges::iota_view{9, 10}});
    // 11 tiles (degenerate)
    check(std::vector{std::ranges::iota_view{0,  1},
                      std::ranges::iota_view{1,  2},
                      std::ranges::iota_view{2,  3},
                      std::ranges::iota_view{3,  4},
                      std::ranges::iota_view{4,  5},
                      std::ranges::iota_view{5,  6},
                      std::ranges::iota_view{6,  7},
                      std::ranges::iota_view{7,  8},
                      std::ranges::iota_view{8,  9},
                      std::ranges::iota_view{9, 10}}, {11});
}

TEST(scan, execution_policy_from_tile_size) {
    arb::threading::task_system ts(1);

    auto check = [&ts](auto&& expected, std::size_t s, int bound = 10) {
        auto r = std::ranges::iota_view{0, bound};
        auto p = arb::threading::execution_policy::create_from_tile_size(ts, s);
        auto t = p.tile(r);
        EXPECT_EQ(t.num_tiles(), expected.size());
        for (std::size_t i=0; auto& r : expected) {
            EXPECT_TRUE(std::ranges::equal(r, t[i++]));
        }
    };
    // degenerate case: zero size
    check(std::vector{std::ranges::iota_view{0,  1},
                      std::ranges::iota_view{1,  2},
                      std::ranges::iota_view{2,  3},
                      std::ranges::iota_view{3,  4},
                      std::ranges::iota_view{4,  5},
                      std::ranges::iota_view{5,  6},
                      std::ranges::iota_view{6,  7},
                      std::ranges::iota_view{7,  8},
                      std::ranges::iota_view{8,  9},
                      std::ranges::iota_view{9, 10}}, 0);
    // degenerate case: empty range
    check(std::vector{std::ranges::iota_view{0, 0}}, 1, 0);
    // degenerate case: zero size, empty range
    check(std::vector{std::ranges::iota_view{0, 0}}, 0, 0);
    // degenerate case: size greater than range size
    check(std::vector{std::ranges::iota_view{0, 1}}, 2, 1);
    // tile size = 1
    check(std::vector{std::ranges::iota_view{0,  1},
                      std::ranges::iota_view{1,  2},
                      std::ranges::iota_view{2,  3},
                      std::ranges::iota_view{3,  4},
                      std::ranges::iota_view{4,  5},
                      std::ranges::iota_view{5,  6},
                      std::ranges::iota_view{6,  7},
                      std::ranges::iota_view{7,  8},
                      std::ranges::iota_view{8,  9},
                      std::ranges::iota_view{9, 10}}, 1);
    // tile size = 2
    check(std::vector{std::ranges::iota_view{0,  2},
                      std::ranges::iota_view{2,  4},
                      std::ranges::iota_view{4,  6},
                      std::ranges::iota_view{6,  8},
                      std::ranges::iota_view{8, 10}}, 2);
    // tile size = 3
    check(std::vector{std::ranges::iota_view{0,  3},
                      std::ranges::iota_view{3,  6},
                      std::ranges::iota_view{6,  8},
                      std::ranges::iota_view{8, 10}}, 3);
    // tile size = 4
    check(std::vector{std::ranges::iota_view{0,  4},
                      std::ranges::iota_view{4,  7},
                      std::ranges::iota_view{7, 10}}, 4);
    // tile size = 5
    check(std::vector{std::ranges::iota_view{0,  5},
                      std::ranges::iota_view{5, 10}}, 5);
    // tile size = 6
    check(std::vector{std::ranges::iota_view{0,  5},
                      std::ranges::iota_view{5, 10}}, 6);
    // tile size = 7
    check(std::vector{std::ranges::iota_view{0,  5},
                      std::ranges::iota_view{5, 10}}, 7);
    // tile size = 8
    check(std::vector{std::ranges::iota_view{0,  5},
                      std::ranges::iota_view{5, 10}}, 8);
    // tile size = 9
    check(std::vector{std::ranges::iota_view{0,  5},
                      std::ranges::iota_view{5, 10}}, 9);
    // tile size = 10
    check(std::vector{std::ranges::iota_view{0,  10}}, 10);
    // tile size = 11
    check(std::vector{std::ranges::iota_view{0,  10}}, 11);
}


TEST(scan, test_simple) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        arb::threading::task_system ts(nthreads);
        auto p = arb::threading::execution_policy::create(ts);

        std::vector<int> v(512, 1);

        arb::threading::inclusive_scan(p, v, [](auto const& l, auto const& r) noexcept {return l+r;}, 0);

        for (auto& x : v) {
            std::cout << x << ", ";
        }
        std::cout << std::endl;

        for (int i=0; i<v.size(); ++i) {
            EXPECT_EQ(v[i], i+1);
        }

    }
}

TEST(scan, test_without_init) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
        arb::threading::task_system ts(nthreads);
        auto p = arb::threading::execution_policy::create(ts);

        std::vector<int> v(512, 1);
        arb::threading::inclusive_scan(p, v, [](auto const& l, auto const& r) noexcept {return l+r;});

        for (auto& x : v) {
            std::cout << x << ", ";
        }
        std::cout << std::endl;

        for (int i=0; i<v.size(); ++i) {
            EXPECT_EQ(v[i], i+1);
        }
    }
}

TEST(scan, test_chunk) {
    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
    //for (int nthreads = 1; nthreads < 4; ++nthreads) {
        arb::threading::task_system ts(nthreads);
        auto p = arb::threading::execution_policy::create(ts);

        std::vector v{1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,9,9,9,9,9};
        arb::threading::chunk_by_output_range<decltype(v)> out;
        //using range_t = std::remove_cvref_t<decltype(std::ranges::subrange(std::ranges::begin(v), std::ranges::end(v)))>;
        //std::vector<range_t> out;

        //std::cout << "               ";
        //for (auto x : v) std::cout << x << "  ";
        //std::cout << "\n";
        auto res = arb::threading::chunk_by(p, v, out, [](auto const& l, auto const& r){ return l==r; });


        for (auto r : res) {
            std::cout << "range: ";
                for (auto const & x : r)
                    std::cout << x << ", ";
            std::cout << "\n";
        }
    }
}

//struct non_copyable {
//    non_copyable() = delete;
//    non_copyable(int i) : m{i} {};
//    non_copyable(non_copyable const&) = delete;
//    non_copyable(non_copyable && other) : m{other.m} {}
//
//    int m;
//};
//
//TEST(scan, test_non_copyable) {
//    for (int nthreads = 1; nthreads < 20; nthreads*=2) {
//        arb::threading::task_system ts(nthreads);
//
//        auto view = std::views::iota(0, 512) | std::views::transform([](auto&& i)->non_copyable {return {i};});
//        std::vector<non_copyable> v{view.begin(), view.end()};
//
//        arb::threading::inclusive_scan(ts, v, [](auto const& l, auto const& r) noexcept -> non_copyable {return {l.m+r.m};}, non_copyable{0});
//    }
//}
