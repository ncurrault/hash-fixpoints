# Background

A Merkel Tree is is structure where the each node is labelled with
a hash obtained by concatenating all of its children's hashes.[1] In a git
repo, there is a very specific sequence of bytes used in each layer of the
tree.[2][3] But essentially it is a specific variant of a Merkel tree.

This project (both the CPU demo and its CUDA parallelization) will be split
into two parts: one just finds fixpoints of SHA-1. That is, what sequence of
bytes of a given length (constants are currently set to 4 bytes) has a SHA-1
hash that begins with those bytes? Spoiler alert: c94c3f2e

The git puzzle is more complicated. I want to write some hex characters to a
file and have the resulting git commit hash match these. (See my proposal for
an invalid motivation.) To accomplish this, I have created a structure that
could be used for finding Merkel tree fixpoints -- values of a leaf in a Merkel
Tree that produce the same value as a prefix of the root hash. It is a directory
with one file and two subdirectories:

- digest_bits.txt
    A sequence of 1's and 0's, separated by spaces. For each hash, we have to
    decide whether it should be represented as raw bytes, i.e.
        hashlib.sha1(x).digest()
    or hex-encoded, i.e.
        hashlib.sha1(x).hexdigest()
    Walking from the leaf to the root, a 1 indicates that the hash's bytes
    should be hex-encoded. In a git repo, this only happens immediately before
    forming the commit hash.
- prefixes/ and suffixes/
    each contains files 0.txt, 1.txt, ... (one for each layer)

    These encode what bytes (if any) should be appended before and after the
    hash at each layer. If it is at the beginning or end, an empty file is
    created.

[1] https://en.wikipedia.org/wiki/Merkle_tree
[2] https://gist.github.com/masak/2415865
[3] https://git-scm.com/book/en/v2/Git-Internals-Git-Objects


# This CPU demo

Raw SHA-1 Fixpoints
- find_sha1_fixpoint.cpp (compiles to find_sha1_fixpoint)
  + finds a string of bytes (set by PREFIX_LEN) whose SHA1 hash begins with
    those same bytes.
  + c94c3f2e was found in 441.99 secs on my machine (according to
    "time ./find_sha1_fixpoint")

Git Commits
- parse_target_repo.py
  from a given git repo, outputs a directory of files that describes a Merkel
  tree whose root is the commit hash and a leaf of which is where the fixpoint
  should be written (described above).

  This should be run from the root of a git repo with command-line arguments:
  + the file where the commit hash should be written ("dir/subdir/DESIGNDOC")
  + the text whose first occurrence should be replaced with the hash ("...")
  + the desired path of output directory ("../out/"). This directory should not
    exist ahead of time

  Additionally, constants at the top of the script allow customization:
  + commit message, timestamp, and timezone information
    * Because of the timestamp constraint, you can either compute a future
      timestamp and be fast at hitting "enter" with the commit command prepped
      or set GIT_COMMITTER_DATE and GIT_AUTHOR_DATE as described in
      parse_target_repo.py
  + the length of the matching prefix that will be written

- find_git_fixpoint.cpp (compiles to find_git_fixpoint)
  reads directory exported by parse_target_repo.py and searches for a hash
  that, when written in the proper place, causes the commit hash to have the
  proper prefix.
  + this has some endianness issues that are noted, but I could only test on a
    little-endian system

# How to run, Example

An example is included: from test_repo, the following was run:
> python ../parse_target_repo.py dir/subdir/DESIGNDOC ...
    ../test_repo_tree_encoded

Fixpoints can be found by running:
> make
> ./find_tree_fixpoint test_repo_tree_encoded/

On my machine, this found 4c473a (which I verified indeed works) in ~40 seconds.
Running a prefix-length 8 (i.e. 4 bytes, 8 hex digits) example would have taken
about 12 hours to perform an exhaustive search.
