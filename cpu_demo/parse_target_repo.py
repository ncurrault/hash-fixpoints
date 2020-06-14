#! /usr/bin/python3

# run this from the root of a git repository with the command-line arguments
# described in the usage statement below

import sys
import subprocess
import os

AUTHOR = "Nicholas Currault <nicholas.curr@gmail.com>"
TIMEZONE = "-0700"
DESIRED_COMMIT_MESSAGE = "added self-referential commit hash using magic"
DESIRED_COMMIT_TIMESTAMP = "1591753853"
# timestamp (formatted as seconds since UNIX epoch)
# to get git to make a commit at the right time, run the following before
# git commit:
#
# export GIT_COMMITTER_DATE='<timestamp>'
# export GIT_AUTHOR_DATE='<timestamp>'

# values for SHA-1
DIGEST_LEN = 20
HEXDIGEST_LEN = 40
PREFIX_LEN = 6

if len(sys.argv) != 4:
    print("usage: parse_target_repo.py (path to target file) (text to replace with hash) (output directory)")
    sys.exit(1)

target_file = sys.argv[1]
to_replace = bytes(sys.argv[2], encoding="utf-8")
out_dir = sys.argv[3]

dir_layers = [None] + \
    [ bytes(l, encoding="utf-8") for l in target_file.split("/") ]

print("reading relevant hashes from git...")

hashes = [ subprocess.check_output(["git", "rev-parse", "HEAD"])[:-1] ]

for i, layer in enumerate(dir_layers):
    curr_tree = subprocess.check_output(["git", "cat-file", "-p", hashes[-1]])

    if i == 0:
        hash = curr_tree[5: 5 + HEXDIGEST_LEN]
    else:
        hash_end = curr_tree.find(b"\t%s\n" % layer)
        hash_start = hash_end - HEXDIGEST_LEN
        hash = curr_tree[hash_start:hash_end]

    hashes.append(hash)

print("reading relevant objects from .git/objects...")

hashes = hashes[::-1]
# reverse order of hashes so the blob we are writing to is first in the output

merkle_layer_prefixes = []
merkle_layer_suffixes = []

# Git stores the file tree in a Merkle Tree (the root of a tree where each
# parent is the SHA-1 hash of its children's hashes in a certain format)

digest_types = [False for _ in range(len(hashes) - 2)] + [True, False]
# depending on the point, Git either feeds the bytes in direcly or
# (for the commit) feeds in a hexadecimal string
# True = hexdigest
# False = digest

for i in range(len(hashes)):
    hash = hashes[i].decode()
    with open(".git/objects/{}/{}".format(hash[:2], hash[2:]), "rb") as f:
        git_obj_contents = subprocess.check_output(["pigz", "-d"], stdin=f)
        # NOTE this is a program that decodes git object files
        # https://stackoverflow.com/a/3178638

    if i == 0:
        prefix_end = git_obj_contents.find(to_replace)
        suffix_begin = prefix_end + len(to_replace)
    else:
        if digest_types[i - 1]:
            prev_hash = bytes(prev_hash, encoding='utf-8')
        else:
            prev_hash = bytes.fromhex(prev_hash)
        prefix_end = git_obj_contents.find(prev_hash)
        suffix_begin = prefix_end + \
            (HEXDIGEST_LEN if digest_types[i - 1] else DIGEST_LEN)

    merkle_layer_prefixes.append(git_obj_contents[:prefix_end])
    merkle_layer_suffixes.append(git_obj_contents[suffix_begin:])
    prev_hash = hash

# overwrite the final layer (above code copied commit bytes from HEAD)
commit_suffix = bytes("""
parent {parent_commit}
author {author_str} {timestamp} {timezone}
committer {author_str} {timestamp} {timezone}

{commit_message}
""".format(parent_commit=hashes[-1].decode(), author_str=AUTHOR,
    timestamp=DESIRED_COMMIT_TIMESTAMP, timezone=TIMEZONE,
    commit_message=DESIRED_COMMIT_MESSAGE), encoding="utf-8")
commit_prefix = bytes("commit {}\0tree ".format(
    len(commit_suffix) + 5 + HEXDIGEST_LEN), encoding="utf-8")
    # total size is suffix + tree hash + len("tree ")

merkle_layer_prefixes[-1] = commit_prefix
merkle_layer_suffixes[-1] = commit_suffix

# ensure blob header is accurate with prefix length
merkle_layer_prefixes[0] = merkle_layer_prefixes[0][ merkle_layer_prefixes[0].find(b"\0") + 1:]
actual_size = len(merkle_layer_prefixes[0]) + len(merkle_layer_suffixes[0]) + PREFIX_LEN
merkle_layer_prefixes[0] = (b"blob %d\0" % actual_size) + merkle_layer_prefixes[0]

print("saving bytes to directory...")
os.makedirs(out_dir + "/prefixes")
os.makedirs(out_dir + "/suffixes")

i = 0
for prefix, suffix, digest_type in zip(merkle_layer_prefixes, merkle_layer_suffixes, digest_types):
    with open("{}/prefixes/{}.txt".format(out_dir, i), "wb") as f:
        f.write(prefix)
    with open("{}/suffixes/{}.txt".format(out_dir, i), "wb") as f:
        f.write(suffix)
    i += 1

with open(out_dir + "/digest_bits.txt", "a") as f:
    f.write(" ".join(map(lambda b: str(int(b)), digest_types)))
