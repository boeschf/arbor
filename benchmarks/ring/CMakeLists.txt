set(ARB_BENCH_RING_DIR ${ARB_BENCH_DIR}/ring)
set(ARB_BENCH_SCRATCH_RING_DIR ${ARB_BENCH_SCRATCH_DIR}/ring)

add_custom_target(
    bench_ring
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_RING_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_SCRATCH_RING_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ring> ${ARB_BENCH_RING_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../platforms ${ARB_BENCH_RING_DIR}/platforms
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/platforms ${ARB_BENCH_RING_DIR}/platforms
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.in ${ARB_BENCH_RING_DIR}
    COMMAND ${CMAKE_COMMAND}
        -DINFILE=${CMAKE_CURRENT_SOURCE_DIR}/bench.xml
        -DOUTFILE=${ARB_BENCH_RING_DIR}/bench.xml
        -DOUT_DIR=runs
        -DWORK_DIR=${ARB_BENCH_SCRATCH_RING_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/configure_bench.cmake
)
add_dependencies(benchmarks bench_ring)
add_dependencies(bench_ring ring)

