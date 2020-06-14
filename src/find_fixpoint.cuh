#ifndef CUDA_FIND_FIXPOINT_CUH
#define CUDA_FIND_FIXPOINT_CUH

#include <stdint.h>

#define PREFIX_LEN 4
#define PREFIX_COUNTER_TYPE uint32_t
#define PREFIX_COUNTER_MAX UINT32_MAX

#define HEXDIGEST_LEN 40
#define DIGEST_LEN 20

union PrefixCounter {
    PREFIX_COUNTER_TYPE n;
    uint8_t prefix[PREFIX_LEN];
};

struct TreeData {
    int num_layers;

    int* layer_sizes;
    uint8_t** layer_templates;
    int* insertion_offsets;
    int* insertion_sizes;

    bool* digest_types;
};


void cudaCallShaFixpointSearchKernel(const unsigned int blocks,
    const unsigned int threads_per_block, bool* success, uint8_t* result);

// void cudaTreeFixpointSearchKernel

#endif
