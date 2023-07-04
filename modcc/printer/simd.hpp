#pragma once

constexpr unsigned no_size = unsigned(-1);

struct simd_spec {
    enum simd_abi { none, neon, avx, avx2, avx512, sve, vls_sve, native, default_abi } abi = none;
    //static constexpr unsigned default_width = 2;
    unsigned size = no_size; // -1 => use `simd::native_width` to determine.
    //unsigned width = no_size;

    simd_spec() = default;
    simd_spec(enum simd_abi a, unsigned s = no_size):
        abi(a), size(s)//, width(w==no_size?default_width:w)
    {
        //if (width==no_size) {
        //    // Pick a width based on abi, if applicable.
        //    switch (abi) {
        //    case avx:
        //    case avx2:
        //        width = 4;
        //        break;
        //    case avx512:
        //        width = 8;
        //        break;
        //    case neon:
        //        width = 2;
        //        break;
        //    case vls_sve:
        //        width = 2; // TODO: get correct value
        //        break;
        //    default: ;
        //    }
        ////}

        //switch (abi) {
        //case avx:
        //case avx2:
        //    size = 4;
        //    break;
        //case avx512:
        //    size = 8;
        //    break;
        //case neon:
        //    size = 2;
        //    break;
        //case vls_sve:
        //    size = 2; // TODO: get correct value
        //    break;
        //default: ;
        //}

    }
};
