#include <cstdio>

#include <cuda_runtime.h>

#include "find_fixpoint.cuh"

CUDA_CALLABLE
uint32_t leftrotate(uint32_t a, uint32_t b) {
    uint32_t high = a << b;
    uint32_t low  = a >> (32 - b);
    return high | low;
}

/* adapted from https://en.wikipedia.org/wiki/SHA-1#SHA-1_pseudocode
   assumes message is PREFIX_LEN bytes
*/
CUDA_CALLABLE
void sha1ofPrefix(uint8_t* result, uint8_t* prefix) {
    uint32_t
        h0 = 0x67452301,
        h1 = 0xEFCDAB89,
        h2 = 0x98BADCFE,
        h3 = 0x10325476,
        h4 = 0xC3D2E1F0;


    uint64_t m1 = 8 * PREFIX_LEN;

    // ASSUMPTION: 0 < PREFIX_LEN < 56
    uint8_t message_padded[64];
    memcpy(message_padded, prefix, PREFIX_LEN);
    message_padded[PREFIX_LEN] = 0x80;

    for (int i = 0; i < 8; i++) {
        message_padded[56 + i] = ((uint8_t*)&m1)[ 7 - i ];
    }

    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        uint8_t* current_word = (uint8_t*)(w + i);
        for (int byte = 0; byte < 4; byte++) {
            current_word[3 - byte] = message_padded[ chunk + (4 * i) + byte ];
        }
    }

    for (int i = 16; i < 80; i++) {
        w[i] = leftrotate(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    }

    uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f, k;
    for (int i = 0; i < 80; i++) {
        if (i < 20) {
            f = (b & c) | ((~b) & d);
            k = 0x5A827999;
        } else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        } else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        } else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }

        uint32_t temp = leftrotate(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = leftrotate(b, 30);
        b = a;
        a = temp;
    }

    h0 += a;
    h1 += b;
    h2 += c;
    h3 += d;
    h4 += e;

    for (int i = 0; i < 4; i++) {
        result[ 3 - i] = h0 >> 8 * i;
        result[ 7 - i] = h1 >> 8 * i;
        result[11 - i] = h2 >> 8 * i;
        result[15 - i] = h3 >> 8 * i;
        result[19 - i] = h4 >> 8 * i;
    }
}

__global__
void cudaShaFixpointSearchKernel(bool* success, uint8_t* prefix) {
    PrefixCounter p;
    p.n = blockDim.x * blockIdx.x + threadIdx.x;
    uint8_t result[PREFIX_LEN];

    while (p.n <= PREFIX_COUNTER_MAX) {
        sha1ofPrefix(result, p.prefix);

        if (! memcmp(result, p.prefix, PREFIX_LEN)) {
            *success = true;
            *prefix = p.prefix;
            // TODO quit all threads
        }

        p.n += blockDim.x * gridDim.x;
    }
}


bool cudaCallShaFixpointSearchKernel(const unsigned int blocks,
    const unsigned int threads_per_block, uint8_t* resultDest) {

    bool* success;
    cudaMalloc(&success, sizeof(bool));

    uint8_t* prefix;
    cudaMalloc(&prefix, PREFIX_LEN * sizeof(uint8_t));

    cudaShaFixpointSearchKernel<<<blocks, threads_per_block>>>(success, prefix);

    bool host_success;
    cudaMemcpy(&host_success, success, cudaMemcpyDeviceToHost);
    cudaMemcpy(&resultDest, prefix, cudaMemcpyDeviceToHost);

    return host_success;
}


// TODO
// __global__ void cudaTreeFixpointSearchKernel
// void cudaCall...
