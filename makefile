CC := g++

INCL := -I ../helpfunctions

DEBUG := -g -Wall -Wextra -Wfloat-equal -Wshadow -Wunreachable-code -fsanitize=address -fsanitize=undefined -fsanitize=pointer-compare -Wunused-but-set-variable

CFLAGS := $(INCL) -std=c++17 $(DEBUG)

cfiles :=

hfiles :=


all: final test

final:
	$(CC) $(CFLAGS) $(cfiles) -o build/$@

test:

clean:
	rm -f build/*

