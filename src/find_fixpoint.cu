#include <cstdio>

#include <cuda_runtime.h>

#include "find_fixpoint.cuh"

__device__
uint32_t leftrotate(uint32_t a, uint32_t b) {
    uint32_t high = a << b;
    uint32_t low  = a >> (32 - b);
    return high | low;
}

/* adapted from https://en.wikipedia.org/wiki/SHA-1#SHA-1_pseudocode
   assumes message is PREFIX_LEN bytes
*/
__device__
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
    memset(message_padded, 0, sizeof(message_padded));
    memcpy(message_padded, prefix, PREFIX_LEN);
    message_padded[PREFIX_LEN] = 0x80;

    for (int i = 0; i < 8; i++) {
        message_padded[56 + i] = ((uint8_t*)&m1)[ 7 - i ];
    }

    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        uint8_t* current_word = (uint8_t*)(w + i);
        for (int byte = 0; byte < 4; byte++) {
            current_word[3 - byte] = message_padded[ (4 * i) + byte ];
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

__device__ bool arr_equal(uint8_t* a, uint8_t* b, unsigned int size) {
    for (int i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

__global__
void cudaShaFixpointSearchKernel(bool* success, uint8_t* prefix) {
    PrefixCounter p;
    p.n = blockDim.x * blockIdx.x + threadIdx.x;
    uint8_t result[PREFIX_LEN];

    uint prev_n;
    do {
        sha1ofPrefix(result, p.prefix);

        if (arr_equal(result, p.prefix, PREFIX_LEN)) {
            *success = true;
            memcpy(prefix, p.prefix, PREFIX_LEN);
            break;
        } else if (*success) { // another thread found a fixpoint
            break;
        }

        prev_n = p.n;
        p.n += blockDim.x * gridDim.x;
    } while (p.n > prev_n);
}


void cudaCallShaFixpointSearchKernel(const unsigned int blocks,
    const unsigned int threads_per_block, bool* success, uint8_t* result) {

    cudaShaFixpointSearchKernel<<<blocks, threads_per_block>>>(success, result);
}


/***** TREE-SPECIFIC CODE *****/
__device__
void sha1WithInsertion(uint8_t* result, uint8_t* message, uint n_bytes,
    uint8_t* message_insert, uint insert_offset, uint insert_size) {
    uint32_t
        h0 = 0x67452301,
        h1 = 0xEFCDAB89,
        h2 = 0x98BADCFE,
        h3 = 0x10325476,
        h4 = 0xC3D2E1F0;

    uint32_t w[80];
    for (int chunk = 0; chunk < n_bytes; chunk += 64) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            uint8_t* current_word = (uint8_t*)(w + i);
            #pragma unroll
            for (int byte = 0; byte < 4; byte++) {
                int read_idx = chunk + (4 * i) + byte;
                current_word[3 - byte] =
                    (read_idx >= insert_offset && read_idx <
                    insert_offset + insert_size)
                    ? message_insert[read_idx - insert_offset]
                    : message[read_idx];
            }
        }
        // generating the words is also theoretically easier on a big-endian
        // system: memcpy(w, message_padded + chunk, 16 * sizeof(uint32_t));

        #pragma unroll
        for (int i = 16; i < 80; i++) {
            w[i] = leftrotate(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        }

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f, k, temp;

        #pragma unroll
        for (int i = 0; i < 20; i++) {
            f = (b & c) | ((~b) & d);
            k = 0x5A827999;

            temp = leftrotate(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = leftrotate(b, 30);
            b = a;
            a = temp;
        }
        #pragma unroll
        for (int i = 20; i < 40; i++) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;

            temp = leftrotate(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = leftrotate(b, 30);
            b = a;
            a = temp;
        }
        #pragma unroll
        for (int i = 40; i < 60; i++) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;

            temp = leftrotate(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = leftrotate(b, 30);
            b = a;
            a = temp;
        }
        #pragma unroll
        for (int i = 60; i < 80; i++) {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;

            temp = leftrotate(a, 5) + f + e + k + w[i];
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
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        result[ 3 - i] = h0 >> 8 * i;
        result[ 7 - i] = h1 >> 8 * i;
        result[11 - i] = h2 >> 8 * i;
        result[15 - i] = h3 >> 8 * i;
        result[19 - i] = h4 >> 8 * i;
    }
}

__device__
void hex_digest_inplace(uint8_t* arr) {
    const uint8_t digits[] = "0123456789abcdef";

    char out[HEXDIGEST_LEN + 1]; // extra space for null terminator
    for (int i = 0; i < HEXDIGEST_LEN >> 1; i++) {
        out[2 * i] = digits[arr[i] >> 4];
        out[2 * i + 1] = digits[arr[i] & 0x0f];
    }
    memcpy(arr, out, HEXDIGEST_LEN);
}

__global__
void cudaTreeFixpointSearchKernel(bool* success, uint8_t* return_prefix,
    struct TreeData* tree) {

    uint8_t hash[HEXDIGEST_LEN];

    PrefixCounter p;
    p.n = blockDim.x * blockIdx.x + threadIdx.x;
    uint prev_n;
    do {
        memcpy(hash, p.prefix, PREFIX_LEN);
        hex_digest_inplace(hash);

        for (int layer = 0; layer < tree->num_layers; layer++) {
            sha1WithInsertion(hash, tree->layer_templates[layer],
                tree->layer_sizes[layer], hash, tree->insertion_offsets[layer],
                tree->insertion_sizes[layer]);

            if (tree->digest_types[layer]) {
                hex_digest_inplace(hash);
            }
        }

        if (arr_equal(hash, p.prefix, PREFIX_LEN)) {
            *success = true;
            memcpy(return_prefix, p.prefix, PREFIX_LEN);
            break;
        } else if (*success) { // another thread found a fixpoint
            break;
        }

        prev_n = p.n;
        p.n += blockDim.x * gridDim.x;
    } while (p.n > prev_n);
}


void cudaCallTreeFixpointSearchKernel(const unsigned int blocks,
    const unsigned int threads_per_block, bool* success, uint8_t* result,
    struct TreeData* tree) {

    cudaTreeFixpointSearchKernel
        <<<blocks, threads_per_block>>>
        (success, result, tree);
}
