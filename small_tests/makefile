CC := g++

#INCL := -I ../helpfunctions

DEBUG := -g -Wall -Wextra -Wfloat-equal -Wshadow -Wunreachable-code -fsanitize=address -fsanitize=undefined -fsanitize=pointer-compare -Wunused-but-set-variable -pedantic

CFLAGS := $(INCL) -std=c++17 -DVERSION=$(BASEVERSION)

all: householder

householder: test_householder.cc test_householder.hh
	$(CC) $(CFLAGS) $(DEBUG) test_householder.cc -o $@
