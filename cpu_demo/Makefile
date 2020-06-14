CC = g++
FLAGS = -g -O2 -Wall

all: find_sha1_fixpoint find_tree_fixpoint

find_sha1_fixpoint: find_sha1_fixpoint.cpp
	$(CC) $(FLAGS) -o find_sha1_fixpoint find_sha1_fixpoint.cpp

find_tree_fixpoint: find_tree_fixpoint.cpp
	$(CC) $(FLAGS) -o find_tree_fixpoint find_tree_fixpoint.cpp

clean:
	rm -rf find_sha1_fixpoint find_tree_fixpoint *.o *.dSYM

.PHONY: clean
