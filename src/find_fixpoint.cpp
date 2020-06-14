#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>


#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#include "find_fixpoint.cuh"

#include "ta_utilities.hpp"

using std::cerr;
using std::cout;
using std::endl;

/*
Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


void check_args(int argc, char **argv){

#if SIMPLE_SHA
    if (argc != 3){
        std::cerr << "Incorrect number of arguments.\n";
        std::cerr << "Arguments: <threads per block> <max number of blocks>\n";
        exit(EXIT_FAILURE);
    }
#else
    if (argc != 4){
        std::cerr << "Incorrect number of arguments.\n";
        std::cerr << "Arguments: <threads per block> <max number of blocks> "
            << "<tree directory>\n";
        exit(EXIT_FAILURE);
    }
#endif
}

void print_hex(uint8_t* arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%02x", arr[i]);
    }
}

int sha1_main(int argc, char **argv){
    uint8_t result[PREFIX_LEN];

    const unsigned int threads_per_block = atoi(argv[1]);
    const unsigned int max_blocks = atoi(argv[2]);

    bool h_success, *d_success;
    gpuErrchk(cudaMalloc(&d_success, sizeof(bool)));
    gpuErrchk(cudaMemset(d_success, 0, sizeof(bool)));

    uint8_t *d_result;
    gpuErrchk(cudaMalloc(&d_result, PREFIX_LEN * sizeof(uint8_t)));

    cudaCallShaFixpointSearchKernel(max_blocks, threads_per_block, d_success, d_result);

    gpuErrchk(cudaMemcpy(&h_success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(result, d_result, PREFIX_LEN * sizeof(uint8_t),
        cudaMemcpyDeviceToHost));

    if (h_success) {
        print_hex(result, PREFIX_LEN);
        std::cout << " is a fixpoint\n";
    } else {
        std::cout << "no fixpoints found :(\n";
    }

    return EXIT_SUCCESS;
}

/***** BEGINNING OF TREE-RELATED CODE *****/

// see TreeData struct in find_fixpoint.cuh

// https://stackoverflow.com/a/5840160
std::ifstream::pos_type filesize(const char* filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}
void load_tree_from_dir(struct TreeData& tree, const char* dir) {
    // if you're reading this function I'm sorry

    // STEP 1: load affix data from the filesystem
    int num_layers;
    bool* digest_types;

    int* prefix_sizes;
    char** prefixes;

    int* suffix_sizes;
    char** suffixes;

    char* digest_bits_fname = (char*) malloc(strlen(dir) + 20);
    strcpy(digest_bits_fname, dir);
    strcat(digest_bits_fname, "digest_bits.txt");
    std::ifstream digest_bits_stream(digest_bits_fname);
    num_layers = 0;
    int capacity = 8;
    digest_types = (bool *) malloc(capacity * sizeof(bool));
    char bit;
    while (digest_bits_stream >> bit) {
        num_layers++;
        if (num_layers > capacity) {
            capacity <<= 1;
            digest_types = (bool*) realloc(digest_types,
                capacity * sizeof(bool));
        }
        if (bit == '1') {
            digest_types[num_layers - 1] = true;
        } else if (bit == '0') {
            digest_types[num_layers - 1] = false;
        } else {
            std::cerr << "unexpected bit in digest_bits.txt: " << bit << "\n";
            exit(1);
        }
    }
    free(digest_bits_fname);
    assert(num_layers > 0);

    prefix_sizes = (int *) malloc(num_layers * sizeof(int));
    prefixes = (char**) malloc(num_layers * sizeof(char*));

    suffix_sizes = (int *) malloc(num_layers * sizeof(int));
    suffixes = (char**) malloc(num_layers * sizeof(char*));

    for (int i = 0; i < num_layers; i++) {
        char prefix_fname[50];
        char suffix_fname[50];
        sprintf(prefix_fname, "%s/prefixes/%d.txt", dir, i);
        sprintf(suffix_fname, "%s/suffixes/%d.txt", dir, i);

        prefix_sizes[i] = filesize(prefix_fname);
        suffix_sizes[i] = filesize(suffix_fname);

        prefixes[i] = (char*) malloc(prefix_sizes[i]);
        suffixes[i] = (char*) malloc(suffix_sizes[i]);

        std::ifstream prefix_in, suffix_in;
        prefix_in.open(prefix_fname);
        suffix_in.open(suffix_fname);

        prefix_in.read(prefixes[i], prefix_sizes[i]);
        suffix_in.read(suffixes[i], suffix_sizes[i]);

        prefix_in.close();
        suffix_in.close();
    }

    // STEP 2: translate affix data into TreeData (essentially construct each
    // layer's data with a hole for the hash that's the proper size)
    tree.num_layers = num_layers;
    tree.insertion_offsets = prefix_sizes;
    // TODO free everything else

    tree.layer_sizes = (int*) malloc(num_layers * sizeof(int));
    tree.layer_templates = (uint8_t**)
        malloc(num_layers * sizeof(uint8_t*));
    tree.insertion_sizes = (int*) malloc(num_layers * sizeof(int));

    tree.digest_types = digest_types;

    for (int layer = 0; layer < num_layers; layer++) {
        if (layer == 0) {
            tree.insertion_sizes[layer] = 2 * PREFIX_LEN;
            // NOTE: assumes initial hash written to file is hex-digested
            // TODO: make this customizable in digest_bits.txt instead of the
            // final output (as what is desired there depends on the context)
        } else if (tree.digest_types[layer - 1]) {
            tree.insertion_sizes[layer] = HEXDIGEST_LEN;
        } else {
            tree.insertion_sizes[layer] = DIGEST_LEN;
        }

        int raw_size = prefix_sizes[layer] + suffix_sizes[layer] +
            tree.insertion_sizes[layer];

        // perform SHA-1 preprocessing while loading the data for efficiency
        // (reallocating memory in each iteration/GPU thread is very slow)
        uint64_t m1 = 8 * raw_size;

        int pad_bytes = 56 - (raw_size % 64);
        if (pad_bytes <= 0) { // if = 0, need to increase so there's room for 0x80
            pad_bytes += 64;
        }
        tree.layer_sizes[layer] = raw_size + pad_bytes + 8;

        tree.layer_templates[layer] = (uint8_t*)
            calloc(tree.layer_sizes[layer], sizeof(uint8_t));
            // ensures that memory used for padding is zeroed

        memcpy(tree.layer_templates[layer],
            prefixes[layer], prefix_sizes[layer]);
        memcpy(tree.layer_templates[layer] + prefix_sizes[layer] +
            tree.insertion_sizes[layer], suffixes[layer],
            suffix_sizes[layer]);

        tree.layer_templates[layer][raw_size] = 0x80;
        for (int i = 0; i < 8; i++) {
            // have to reverse because system is litte-endian and SHA-1
            // requires big-endian ordering
            tree.layer_templates[layer][raw_size + pad_bytes + i] =
                ((uint8_t*)&m1)[ 7 - i ];
        }
    }

    for (int i = 0; i < num_layers; i++) {
        free(prefixes[i]);
        free(suffixes[i]);
    }
    free(prefixes);
    free(suffixes);
    free(suffix_sizes);
    // digest_types, prefix_sizes are used in the struct
}

void tree_host_to_device(struct TreeData& source, struct TreeData* dest) {
    struct TreeData tmp;
    // will be copied to device after it is populated with dev pointers

    int num_layers = source.num_layers; // saves space throughout
    tmp.num_layers = num_layers;

    gpuErrchk(cudaMalloc(&tmp.layer_sizes, num_layers * sizeof(int)));
    gpuErrchk(cudaMemcpy(tmp.layer_sizes, source.layer_sizes,
        num_layers * sizeof(int), cudaMemcpyHostToDevice));

    uint8_t** d_layer_templates = (uint8_t**) malloc(num_layers * sizeof(uint8_t*));
    for (int i = 0; i < num_layers; i++) {
        gpuErrchk(cudaMalloc(&d_layer_templates[i], source.layer_sizes[i]));
        gpuErrchk(cudaMemcpy(d_layer_templates[i], source.layer_templates[i],
            source.layer_sizes[i], cudaMemcpyHostToDevice));
    }
    gpuErrchk(cudaMalloc(&tmp.layer_templates, num_layers * sizeof(uint8_t*)));
    gpuErrchk(cudaMemcpy(tmp.layer_templates, d_layer_templates,
        num_layers * sizeof(uint8_t*), cudaMemcpyHostToDevice));
    free(d_layer_templates);

    gpuErrchk(cudaMalloc(&tmp.insertion_offsets, num_layers * sizeof(int)));
    gpuErrchk(cudaMemcpy(tmp.insertion_offsets, source.insertion_offsets,
        num_layers * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&tmp.insertion_sizes, num_layers * sizeof(int)));
    gpuErrchk(cudaMemcpy(tmp.insertion_sizes, source.insertion_sizes,
        num_layers * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&tmp.digest_types, num_layers * sizeof(bool)));
    gpuErrchk(cudaMemcpy(tmp.digest_types, source.digest_types,
        num_layers * sizeof(bool), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(dest, &tmp, sizeof(struct TreeData),
        cudaMemcpyHostToDevice));
}
void free_host_tree_contents(struct TreeData& tree) {
    for (int i = 0; i < tree.num_layers; i++) {
        free(tree.layer_templates[i]);
    }
    free(tree.layer_templates);
    free(tree.layer_sizes);
    free(tree.insertion_offsets);
    free(tree.insertion_sizes);
    free(tree.digest_types);
}
void free_device_tree_contents(struct TreeData* dev_tree) {
    struct TreeData tmp;
    gpuErrchk(cudaMemcpy(&tmp, dev_tree, sizeof(struct TreeData),
        cudaMemcpyDeviceToHost));

    uint8_t** d_layer_templates = (uint8_t**) malloc(num_layers * sizeof(uint8_t*));
    gpuErrchk(cudaMemcpy(d_layer_templates, tmp.layer_templates,
        num_layers * sizeof(uint8_t*), cudaMemcpyDeviceToHost));

    for (int i = 0; i < tmp.num_layers; i++) {
        gpuErrchk(cudaFree(d_layer_templates[i]));
    }
    free(d_layer_templates);

    gpuErrchk(cudaFree(tmp.layer_templates));
    gpuErrchk(cudaFree(tmp.layer_sizes));
    gpuErrchk(cudaFree(tmp.insertion_offsets));
    gpuErrchk(cudaFree(tmp.insertion_sizes));
    gpuErrchk(cudaFree(tmp.digest_types));
}

int tree_main(int argc, char **argv) {
    const unsigned int threads_per_block = atoi(argv[1]);
    const unsigned int max_blocks = atoi(argv[2]);

    struct TreeData tree;
    load_tree_from_dir(tree, argv[3]);

    std::cout << "finished loading data from disk to CPU\n";

    struct TreeData* d_tree;
    gpuErrchk(cudaMalloc(&d_tree, sizeof(struct TreeData)));
    tree_host_to_device(tree, d_tree);
    free_host_tree_contents(tree);

    // the rest is the same as for simple SHA-1:
    bool h_success, *d_success;
    gpuErrchk(cudaMalloc(&d_success, sizeof(bool)));
    gpuErrchk(cudaMemset(d_success, 0, sizeof(bool)));

    uint8_t *d_result;
    uint8_t result[PREFIX_LEN];
    gpuErrchk(cudaMalloc(&d_result, PREFIX_LEN * sizeof(uint8_t)));

    std::cout << "finished loading data from CPU to GPU\n";

    cudaCallTreeFixpointSearchKernel(max_blocks, threads_per_block, d_success,
        d_result, d_tree);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // recommended by the same StackOverflow page cited for gpuErrchk

    gpuErrchk(cudaMemcpy(&h_success, d_success, sizeof(bool),
        cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(result, d_result, PREFIX_LEN * sizeof(uint8_t),
        cudaMemcpyDeviceToHost));

    if (h_success) {
        print_hex(result, PREFIX_LEN);
        std::cout << " is a fixpoint\n";
    } else {
        std::cout << "no fixpoints found :(\n";
    }

    gpuErrchk(cudaFree(d_success));
    gpuErrchk(cudaFree(d_result));
    free_device_tree_contents(d_tree);
    gpuErrchk(cudaFree(d_tree));

    return EXIT_SUCCESS;
}


/***** END OF TREE-RELATED CODE *****/

int main(int argc, char **argv){
    // This project will be tested on a course server; these are left in as a courtesy
    TA_Utilities::select_coldest_GPU();

    int max_time_allowed_in_seconds = 360;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    check_args(argc, argv);

    // NOTE: this task doesn't use any shared memory but it accesses global
    // memory a fair amount, so more L1 cache should be beneficial
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#if SIMPLE_SHA
    return sha1_main(argc, argv);
#else
    return tree_main(argc, argv);
#endif
}
