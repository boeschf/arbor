
#set(BENCH_BUSYRING_BENCHMARK_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/bench_busyring)
#set(BENCH_BUSYRING_OUTPUT_DIR ${BENCH_BUSYRING_BENCHMARK_DIR}/output)

set(ARB_BENCH_BUSYRING_DIR ${ARB_BENCH_DIR}/busyring)
set(ARB_BENCH_SCRATCH_BUSYRING_DIR ${ARB_BENCH_SCRATCH_DIR}/busyring)

#add_custom_target(bench_busyring)
##    #DEPENDS test.xml
##    COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_BUSYRING_DIR}
##    COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_SCRATCH_BUSYRING_DIR}
##    #COMMAND ${CMAKE_COMMAND} -E configure_file test.xml ${BENCH_BUSYRING_BENCHMARK_DIR}/test.xml
##    #COMMAND ${CMAKE_COMMAND} -P configure -DOUTFILE=${BENCH_BUSYRING_BENCHMARK_DIR}/test.xml
##)
#add_dependencies(benchmarks bench_busyring)
#add_dependencies(bench_busyring busyring)
#
#add_custom_command(
#    TARGET bench_busyring
#    PRE_BUILD
#    DEPENDS test.xml
#    COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_BUSYRING_DIR}
#    COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_SCRATCH_BUSYRING_DIR}
#    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:busyring> ${ARB_BENCH_BUSYRING_DIR}
#    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/platforms ${ARB_BENCH_BUSYRING_DIR}/platforms
#    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.in ${ARB_BENCH_BUSYRING_DIR}
#    COMMAND ${CMAKE_COMMAND}
#        -DINFILE=${CMAKE_CURRENT_SOURCE_DIR}/bench.xml
#        -DOUTFILE=${ARB_BENCH_BUSYRING_DIR}/bench.xml
#        -DOUT_DIR=bench_run
#        -DSCRATCH_DIR=${ARB_BENCH_SCRATCH_BUSYRING_DIR}
#        -P ${CMAKE_CURRENT_SOURCE_DIR}/configure.cmake
#)

add_custom_target(
    bench_busyring
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_BUSYRING_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ARB_BENCH_SCRATCH_BUSYRING_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:busyring> ${ARB_BENCH_BUSYRING_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/platforms ${ARB_BENCH_BUSYRING_DIR}/platforms
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.in ${ARB_BENCH_BUSYRING_DIR}
    COMMAND ${CMAKE_COMMAND}
        -DINFILE=${CMAKE_CURRENT_SOURCE_DIR}/bench.xml
        -DOUTFILE=${ARB_BENCH_BUSYRING_DIR}/bench.xml
        -DOUT_DIR=bench_run
        -DWORK_DIR=${ARB_BENCH_SCRATCH_BUSYRING_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/configure_bench.cmake
)
add_dependencies(benchmarks bench_busyring)
add_dependencies(bench_busyring busyring)


##CMAKE_RUNTIME_OUTPUT_DIRECTORY

#configure_file(
#    test.xml
#    ${ARB_BENCH_BUSYRING_DIR}/test.xml
#    @ONLY
#    NEWLINE_STYLE UNIX)

#if (TARGET bench_busyring)
#    message(STATUS "bench_busyring is requested")
#endif()

#{
#    "name": "busy_ring_1",
#    "num-cells": 2000,
#    "ring-size": 10,
#    "duration": 80,
#    "dt": 0.025,
#    "min-delay": 5, 
#    "record": false
#}
