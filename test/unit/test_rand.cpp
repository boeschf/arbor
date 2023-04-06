#include <array>
#include <cstdio>
#include <iostream>

#include <gtest/gtest.h>

#include <Random123/uniform.hpp>
#include "backends/rand_impl.hpp"

#define ARB_TEST_RAND_PRINT 0

constexpr unsigned M = arb::cbprng::array_type::static_size;

#if ARB_TEST_RAND_PRINT
void print_as_hex(unsigned long int v){ std::printf("%016lx\n", v); }
void print_as_hex(double v) { std::cout << std::hexfloat << v << "\n"; }
#else
void print_as_hex(...){ }
#endif

template<typename R, typename E>
bool comp(const R& result, const E& expected) {
    bool ret = true;
    for (unsigned i=0; i<M; ++i) {
        print_as_hex(result[i]);
        ret = ret && (result[i] == expected[i]);
    }
    return ret;
}

bool verify_raw(arb::cbprng::array_type counter, arb::cbprng::array_type key,
    arb::cbprng::array_type expected) {
    const auto r = arb::cbprng::generator{}(counter,key);
    return comp(r, expected);
}

bool verify_uniform(arb::cbprng::array_type raw, std::array<double, 4> expected) {
    const auto r = r123::u01all<double>(raw);
    return comp(r, expected);
}

bool verify_normal(arb::cbprng::array_type raw, std::array<double, 4> expected) {
    const auto [a0, a1] = r123::boxmuller(raw[0], raw[1]);
    const auto [a2, a3] = r123::boxmuller(raw[2], raw[3]);
    return comp(std::array{a0, a1, a2, a3}, expected);
}

struct record {
    arb::cbprng::array_type counter;
    arb::cbprng::array_type key;
    arb::cbprng::array_type expected_raw;
    std::array<double, M> expected_uniform;
    std::array<double, M> expected_normal;
};

std::array records = {
    record{
        {0ul, 0ul, 0ul, 0ul},
        {0ul, 0ul, 0ul, 0ul},
        {0x0068c71d9376b741ul, 0x400933a14e65d6c4ul, 0xeae334bacaeedb8eul, 0x4e8fdcfaedb0c1bbul},
        {0x1.a31c764ddaddp-10, 0x1.0024ce8539976p-2, 0x1.d5c6697595ddbp-1, 0x1.3a3f73ebb6c3p-2},
        {0x1.11fdbe3525a5ap-6, 0x1.aa28fe146c6cap+0, -0x1.85ca13e10e23ep-1, 0x1.55d559c30f48p+0}},
    record{
        {1ul, 2ul, 3ul, 4ul},
        {5ul, 6ul, 7ul, 8ul},
        {0x101d6de5a38c39c3ul ,0x67f50f06803a8d74ul, 0x365901c588daa56aul, 0x08ab764ec508eddeul},
        {0x1.01d6de5a38c3ap-4, 0x1.9fd43c1a00ea3p-2, 0x1.b2c80e2c46d53p-3, 0x1.156ec9d8a11dcp-5},
        {0x1.08d6a9993c6c4p-1, 0x1.3d27864a4c76p+0, 0x1.43c2878d17a67p+1, 0x1.38abb6ec31e01p-1}},
    record{
        {6666ul, 77ul, 500ul, 8363839ul},
        {999ul, 137ul, 0xdeadf00dul, 0xdeadbeeful},
        {0x3396ffbcb3bc62baul, 0xc5e469e108743da9ul, 0x54c5ff79567b21b7ul, 0x7867540de44f143eul},
        {0x1.9cb7fde59de31p-3, 0x1.8bc8d3c210e88p-1, 0x1.5317fde559ec8p-2, 0x1.e19d5037913c5p-2},
        {0x1.5e7c477550d2fp-1, 0x1.b8bbe4d248957p-3, 0x1.12722f2406fc7p+0, -0x1.32ec9d7df4626p-1}}};

// compare generated random integers to pre-recorded values
TEST(cbprng, reproducible_raw) {
    for (const auto& r : records) {
        EXPECT_TRUE(verify_raw(r.counter, r.key, r.expected_raw));
    }
}

// compare generated uniformly distributed values to pre-recorded values
TEST(cbprng, reproducible_uniform) {
    for (const auto& r : records) {
        EXPECT_TRUE(verify_uniform(r.expected_raw, r.expected_uniform));
    }
}

// compare generated normally distributed values to pre-recorded values
TEST(cbprng, reproducible_normal) {
    for (const auto& r : records) {
        EXPECT_TRUE(verify_normal(r.expected_raw, r.expected_normal));
    }
}
