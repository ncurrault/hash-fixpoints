#ifndef CUDA_FIND_FIXPOINT_CUH
#define CUDA_FIND_FIXPOINT_CUH

#include <stdint.h>

#define PREFIX_LEN 4
#define PREFIX_COUNTER_TYPE uint32_t
#define PREFIX_COUNTER_MAX UINT32_MAX

uint32_t leftrotate(uint32_t a, uint32_t b);
void sha1ofPrefix(uint8_t* result, uint8_t* prefix);

union PrefixCounter {
    PREFIX_COUNTER_TYPE n;
    uint8_t prefix[PREFIX_LEN];
};

void cudaShaFixpointSearchKernel(bool* success, uint8_t* prefix);

bool cudaCallShaFixpointSearchKernel(const unsigned int threads_per_block,
    const unsigned int blocks, uint8_t* resultDest);


// void cudaTreeFixpointSearchKernel

#endif
