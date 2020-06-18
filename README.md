
# Finding Hash Fixpoints in SHA-1 and Merkle Trees Thereof

This project uses CUDA to optimize the search for hash fixpoints -- sequences
of bytes whose SHA-1 hash begins with those bytes -- and the related problem
of making a git commit that writes its own hash to a file.

## Motivation
For the meme. Being able to make a self-referential commit is pretty neat and
being able to parallelize hash problems is useful in a variety of applications.
Additionally, this project allowed me to gain a greater familiarity with the
internals of SHA-1.

## CPU Demo
See `cpu_demo/README.txt`.

## CUDA

### Algorithm Overview
First, the data is loaded in from the directory. We first read it as-is:
digest type bits, prefixes (with lengths), and suffixes (with lengths). We then
put these together into a single array for each layer, leaving the proper amount
of space in the middle of the data for the hash of the prior layer and leaving
space after before adding the length in accordance with the preprocessing steps
of SHA-1. Then, we transfer all of this information to the GPU. We divide
potential hashes among the available threads, and each one tries a hash,
replacing the required data and walking up the tree, and checking if it found a
fixpoint by comparing the final value to the input. If successful, it sets a
global flag, causing all other threads to quit before trying any new hashes.

### File Structure

* `src/find_fixpoint.cpp` contains code whose function depends on the executable:
	* For `sha1-fixpoint`, sets up memory and calls the kernel caller on this memory to search for fixpoints of the prefix length defined in `find_fixpoint.cu`.
	* For `tree-fixpoint`, sets up memory, including loading the data from the directory into a `struct TreeData` in device space, and then calls the appropriate kernel caller.
* `src/find_fixpoint.cu` contains the actual CUDA kernels and callers
* `src/find_fixpoint.cuh` describes the caller functions in `src/find_fixpoint.cu` as well as some `#define`s that are relevant throughout and `struct TreeData`.
* `src/ta_utilities.cpp` and `src/ta_utilities.hpp` allow the script to be "kind" when running on a shared server, selecting the coldest GPU to run on and enforcing a time limit

### Notes
The Makefile produces two executables. For both, the first two arguments define
(in order) the number of threads per block and the maximum number of blocks.

`./tree-fixpoint` also takes a third argument -- the directory that encodes the
Merkel tree that we want to solve for a fixpoint in. (See `cpu_demo/README.txt`
for details on how this is formatted.)

### Optimizations

I did some general optimization for the CPU version before transitioning to CUDA.
For example, I tried to minimize memory copying and do all the SHA-1
preprocessing during initialization to avoid repeating work.

For the GPU, I was unable to find any good opportunities for shared memory, as
it is inherent to hashes that even adjacent inputs will have very different
outputs. So I configured CUDA to use some space that would be used for shared
memory to increase the size of the L1 cache. Besides this, I tried to minimize
conditionals that could cause warp divergence and unrolled several loops in the
SHA-1 implementation.

### Analysis
This project was successful: the CPU version found a 6-hex-digit (3-byte)
fixpoint in ~30 seconds on my laptop, while the CUDA version found an
8-hex-digit (4-byte) fixpoint in ~1 minute (despite a 256x increase in the
search space). Due to the unpredictable nature of hashes, it is uncertain
whether I just got lucky, but even in some test cases where no fixpoint was
found, the program never took more than ~5 minutes to run.

Given the test repo's draft commit (enclosed as "test_repo_prefix8"),
the fixpoint `18b57d21` was found in 1.5 minutes.

### Example Output

When the code successfully finds a fixpoint:
```
$ ./tree-fixpoint 512 32767  test_repo_prefix8/
Index of the GPU with the lowest temperature: 1 (0 C)
Time limit for this program set to 360 seconds
finished loading data from disk to CPU
finished loading data from CPU to GPU
18b57d21 is a fixpoint
```

When the code exhausts all possibilities and no fixpoints are found:
```
$ ./tree-fixpoint 512 32767  test_repo_prefix8_nofix/
Index of the GPU with the lowest temperature: 1 (0 C)
Time limit for this program set to 360 seconds
finished loading data from disk to CPU
finished loading data from CPU to GPU
no fixpoints found :(
```
