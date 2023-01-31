CC := g++

BASEVERSION := 1

INCL := -I ../helpfunctions

DEBUG := -g -Wall -Wextra -Wfloat-equal -Wshadow -Wunreachable-code -fsanitize=address -fsanitize=undefined -fsanitize=pointer-compare -Wunused-but-set-variable -pedantic

CFLAGS := $(INCL) -std=c++17 $(DEBUG) -DVERSION=$(BASEVERSION)

cfiles := main.cc test.cc

hfiles := test.hh qr.hh helpfunctions/helpfunctions.hh


all: base test

base: $(cfiles) $(hfiles)
	$(CC) $(CFLAGS) $(cfiles) -o build/$@

test:

clean:
	rm -f build/*

