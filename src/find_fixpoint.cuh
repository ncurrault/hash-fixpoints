#ifndef CUDA_FIND_FIXPOINT_CUH
#define CUDA_FIND_FIXPOINT_CUH

#include <stdint.h>

#define PREFIX_LEN 4
#define PREFIX_COUNTER_TYPE uint32_t
#define PREFIX_COUNTER_MAX UINT32_MAX

union PrefixCounter {
    PREFIX_COUNTER_TYPE n;
    uint8_t prefix[PREFIX_LEN];
};


bool cudaCallShaFixpointSearchKernel(const unsigned int threads_per_block,
    const unsigned int blocks, uint8_t* resultDest);


// void cudaTreeFixpointSearchKernel

#endif
