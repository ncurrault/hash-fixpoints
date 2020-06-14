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

        tree.layer_sizes[layer] = prefix_sizes[layer] +
            suffix_sizes[layer] + tree.insertion_sizes[layer];
        tree.layer_templates[layer] = (uint8_t*)
            malloc(tree.layer_sizes[layer]);

        memcpy(tree.layer_templates[layer],
            prefixes[layer], prefix_sizes[layer]);
        memcpy(tree.layer_templates[layer] + prefix_sizes[layer] +
            tree.insertion_sizes[layer], suffixes[layer],
            suffix_sizes[layer]);
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
void free_tree_contents(struct TreeData& tree) {
    for (int i = 0; i < tree.num_layers; i++) {
        free(tree.layer_templates[i]);
    }
    free(tree.layer_templates);
    free(tree.layer_sizes);
    free(tree.insertion_offsets);
    free(tree.insertion_sizes);
    free(tree.digest_types);
}

int tree_main(int argc, char **argv) {
    const unsigned int threads_per_block = atoi(argv[1]);
    const unsigned int max_blocks = atoi(argv[2]);

    struct TreeData tree;
    load_tree_from_dir(tree, argv[3]);

    // TODO

    return EXIT_SUCCESS;
}


/***** END OF TREE-RELATED CODE *****/

int main(int argc, char **argv){
    // This project will be tested on a course server; these are left in as a courtesy
    TA_Utilities::select_coldest_GPU();

    int max_time_allowed_in_seconds = 90;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    check_args(argc, argv);

#if SIMPLE_SHA
    return sha1_main(argc, argv);
#else
    return tree_main(argc, argv);
#endif
}
