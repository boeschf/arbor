set(ARB_BENCH_BUSYRING_DIR ${ARB_BENCH_DIR}/busyring)
set(ARB_BENCH_SCRATCH_BUSYRING_DIR ${ARB_BENCH_SCRATCH_DIR}/busyring)

add_custom_target(
    bench_busyring
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_BUSYRING_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_SCRATCH_BUSYRING_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:busyring> ${ARB_BENCH_BUSYRING_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/../platforms ${ARB_BENCH_BUSYRING_DIR}/platforms
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/platforms ${ARB_BENCH_BUSYRING_DIR}/platforms
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.in ${ARB_BENCH_BUSYRING_DIR}
    COMMAND ${CMAKE_COMMAND}
        -DINFILE=${CMAKE_CURRENT_SOURCE_DIR}/bench.xml
        -DOUTFILE=${ARB_BENCH_BUSYRING_DIR}/bench.xml
        -DOUT_DIR=runs
        -DWORK_DIR=${ARB_BENCH_SCRATCH_BUSYRING_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/configure_bench.cmake
)
add_dependencies(benchmarks bench_busyring)
add_dependencies(bench_busyring busyring)
