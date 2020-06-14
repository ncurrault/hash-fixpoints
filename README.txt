# Finding Hash Fixpoints in SHA-1 and Merkle Trees Thereof

This project uses CUDA to optimize the search for hash fixpoints -- sequences
of bytes whose SHA-1 hash begins with those bytes -- and the related problem
of making a git commit that writes its own hash to a file.

For running the CPU versions, see cpu_demo/README.txt

For running the CUDA versions:

The Makefile produces two executables. For both, the first two arguments define
(in order) the number of threads per block and the maximum number of blocks.

./tree-fixpoint also takes a third argument -- the director that encodes the
Merkel tree that we want to solve for a fixpoint in. (See cpu_demo/README.txt
for details on how this is formatted.)

This project was successful: the CPU version found a 6-hex-digit (3-byte)
fixpoint in ~30 seconds on my laptop, while the CUDA version found an
8-hex-digit (4-byte) fixpoint in ~1 minute (despite a 256x increase in the
search space). Due to the unpredictable nature of hashes, it is uncertain
whether I just got lucky, but even in some test cases where no fixpoint was
found, the program never took more than ~5 minutes to run.

Given the test repo's draft commit enclosed as cpu_demo/test_repo_tree_prefix8,
the fixpoint 18b57d21 was found in 1.5 minutes.
