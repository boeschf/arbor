
#include <threading/scan.hpp>

#include <benchmark/benchmark.h>

#include <thread>
#include <ranges>
#include <vector>

//template<typename T>
//auto simple_scan() {
//        //state.PauseTiming();
//    std::size_t nthreads = 4;
//    std::size_t scan_size = 1ull << 20;
//    std::size_t num_tiles = 4*32;
//
//    //arb::threading::task_system ts(nthreads, true);
//    arb::threading::task_system ts(nthreads, false);
//
//    std::vector<T> vec(scan_size, T{1});
//        //state.ResumeTiming();
//
//
//    for (std::size_t i=0; i<5000; ++i) {
//        arb::threading::inclusive_scan(
//            //arb::threading::execution_policy::create_from_tile_size(ts, tile_size),
//            arb::threading::execution_policy::create_from_num_tiles(ts, num_tiles),
//            vec,
//            std::plus<T>{}
//        );
//        arb::threading::inclusive_scan(
//            //arb::threading::execution_policy::create_from_tile_size(ts, tile_size),
//            arb::threading::execution_policy::create_from_num_tiles(ts, num_tiles),
//            vec,
//            [](const auto& l, const auto& r) { return r-l; }
//        );
//    }
//    return vec.back();
//}
//
//int main() {
//    int res = simple_scan<double>();
//    return res;
//}


template<typename T>
void test_scan(benchmark::State& state) {
        //state.PauseTiming();
    std::size_t nthreads = state.range(0);
    std::size_t scan_size = state.range(1);
    //std::size_t tile_size = state.range(2);
    std::size_t num_tiles = state.range(2);
    //arb::threading::task_system ts(nthreads, true);
    arb::threading::task_system ts(nthreads, false);


    std::vector<T> vec(scan_size, T{1});
    //state.ResumeTiming();
    for (unsigned i=0; i<10; ++i) {
        arb::threading::inclusive_scan(
            //arb::threading::execution_policy::create_from_tile_size(ts, tile_size),
            arb::threading::execution_policy::create_from_num_tiles(ts, num_tiles),
            vec,
            std::plus<T>{}
        );
        for (auto& x: vec) x = T{1};
    }

    for (auto _ : state) {
    //while(state.KeepRunning()) {
        auto data = vec.data();         // Allow vec.data() to be clobbered. Pass as non-const
        benchmark::DoNotOptimize(data); // lvalue to avoid undesired compiler optimizations

        arb::threading::inclusive_scan(
            //arb::threading::execution_policy::create_from_tile_size(ts, tile_size),
            arb::threading::execution_policy::create_from_num_tiles(ts, num_tiles),
            vec,
            std::plus<T>{}
        );
        state.PauseTiming();
        for (auto& x: vec) x = T{1};
        state.ResumeTiming();
        benchmark::ClobberMemory();
    }
}

template<typename T>
void test_scan_serial(benchmark::State& state) {
    std::size_t scan_size = state.range(0);

    std::vector<T> vec(scan_size, T{1});
    for (unsigned i=0; i<10; ++i) {
        std::inclusive_scan(vec.begin(), vec.end(), vec.begin());
        for (auto& x: vec) x = T{1};
    }

    for (auto _ : state) {
    //while(state.KeepRunning()) {
        auto data = vec.data();         // Allow vec.data() to be clobbered. Pass as non-const
        benchmark::DoNotOptimize(data); // lvalue to avoid undesired compiler optimizations

        std::inclusive_scan(vec.begin(), vec.end(), vec.begin());

        state.PauseTiming();
        for (auto& x: vec) x = T{1};
        state.ResumeTiming();
        benchmark::ClobberMemory();
    }
}

void run_custom_arguments(benchmark::internal::Benchmark* b) {
    for (auto scan_size_s : std::ranges::iota_view<std::size_t, std::size_t>{10u, 26u}) {
        auto scan_size = 1ull << scan_size_s;
        //for (auto tile_size_s: std::ranges::iota_view<std::size_t, std::size_t>{12u, 16u}) {
        //    auto tile_size = 1ull << tile_size_s;
        //    for (auto nthreads : std::ranges::iota_view<std::size_t,
        //            std::size_t>{1u, 12u/*std::thread::hardware_concurrency()*/+1u}) {
        //        if (tile_size > scan_size) continue;
        //        b->Args({nthreads, scan_size, tile_size});

        //    }
        //}
        //
        //for (auto nthreads : std::ranges::iota_view<std::size_t, std::size_t>{1u, 12u/*std::thread::hardware_concurrency()*/+1u}) {
        for (auto nthreads: {1, 8}) {

            //for (auto num_tiles: {nthreads, 8*nthreads, 16*nthreads, 32*nthreads}) {
            {
                auto num_tiles = nthreads*16;
                    //std::ranges::iota_view<std::size_t, std::size_t>{nthreads, nthreads*32u}) {
                //auto tile_size = 1ull << tile_size_s;
                //if (tile_size > scan_size) continue;
                b->Args({nthreads, scan_size, num_tiles});

            }
        }
    }
}

void run_custom_arguments_serial(benchmark::internal::Benchmark* b) {
    for (auto scan_size_s : std::ranges::iota_view<std::size_t, std::size_t>{10u, 26u}) {
        auto scan_size = 1ull << scan_size_s;
        b->Args({scan_size});
    }
}

BENCHMARK(test_scan<double>)->Apply(run_custom_arguments);
BENCHMARK(test_scan_serial<double>)->Apply(run_custom_arguments_serial);

BENCHMARK_MAIN();
