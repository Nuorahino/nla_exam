CC=g++
#CCFLAGS=-fno-trapping-math -fno-math-errno -fno-signed-zeros -fno-builtin -DNDEBUG -msse -msse2 -mavx -O3 -pg
CCFLAGS=-fno-trapping-math -fno-math-errno -fno-signed-zeros -fno-builtin -DNDEBUG -msse -msse2 -mavx -O3 -llapack
INCLUDEDIR=../../include/

main: profile.cc
	$(CC) $(CCFLAGS) -I$(INCLUDEDIR) profile.cc
