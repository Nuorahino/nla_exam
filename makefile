CC := g++

BASEVERSION := 1
#BASEVERSION := 999

INCL := -I ../helpfunctions

DEBUG := -g -Wall -Wextra -Wfloat-equal -Wshadow -Wunreachable-code -fsanitize=address -fsanitize=undefined -fsanitize=pointer-compare -Wunused-but-set-variable -pedantic

CFLAGS := $(INCL) -std=c++17 -llapack $(CPPFLAGS) -DVERSION=$(BASEVERSION)

cfiles := main.cc test.cc test_cases.cc

hfiles := test.hh qr.hh helpfunctions/helpfunctions.hh test_cases.hh


all: debug O1 O2 O3

debug: $(cfiles) $(hfiles)
	$(CC) $(CFLAGS)0 $(DEBUG) $(cfiles) -o build/$@

O1: $(cfiles) $(hfiles)
	$(CC) $(CFLAGS)1 -O1 $(cfiles) -o build/$@

O2: $(cfiles) $(hfiles)
	$(CC) $(CFLAGS)2 -O2 $(cfiles) -o build/$@

O3: $(cfiles) $(hfiles)
	$(CC) $(CFLAGS)3 -O3 $(cfiles) -o build/$@

clean:
	rm -f build/*

