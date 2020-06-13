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

const float PI = 3.14159265358979;

#if AUDIO_ON
    #include <sndfile.h>
#endif


float gaussian(float x, float mean, float std){
    return (1 / (std * sqrt(2 * PI) ) )
        * exp(-1.0/2.0 * pow((x - mean) / std, 2) );
}

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
    if (argc < 5){
        std::cerr << "Incorrect number of arguments.\n";
        std::cerr << "Arguments: <threads per block> <max number of blocks> <path to bytes directory> [digest type bits]\n";
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

    bool success = cudaCallShaFixpointSearchKernel(threads_per_block, max_blocks, result);
    if (success) {
        print_hex(result, PREFIX_LEN);
        std::cout << " is a fixpoint\n";
    } else {
        std::cout << "no fixpoints found :(\n";
    }

    return EXIT_SUCCESS;
}

int tree_main(int argc, char **argv) {
    /* TODO */
    std::cout << "tree_main\n";
    return EXIT_SUCCESS;
}

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
