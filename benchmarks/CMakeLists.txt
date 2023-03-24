
option(ARB_BENCHMARKS "build benchmarks" OFF)

if (ARB_BENCHMARKS)

    set(ARB_BENCH_DIR ${CMAKE_BINARY_DIR}/benchmarking CACHE STRING
        "benchmark directory")

    set(ARB_BENCH_SCRATCH_DIR ${CMAKE_BINARY_DIR}/benchmarking/scratch CACHE STRING
        "benchmark scratch directory")

    add_custom_target(
        benchmarks
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_SCRATCH_DIR})

    cmake_dependent_option(ARB_BENCH_BUSYRING "busyring benchmark" ON ARB_BENCHMARKS OFF)
    cmake_dependent_option(ARB_BENCH_RING "ring benchmark" ON ARB_BENCHMARKS OFF)

    if (ARB_BENCH_BUSYRING)
        add_subdirectory(busyring)
    endif()

    if (ARB_BENCH_RING)
        add_subdirectory(ring)
    endif()

endif()
