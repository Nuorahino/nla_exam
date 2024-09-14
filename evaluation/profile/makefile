CC=g++
#CCFLAGS=-fno-trapping-math -fno-math-errno -fno-signed-zeros -fno-builtin -DNDEBUG -msse -msse2 -mavx -O3 -pg
CCFLAGS=-fno-trapping-math -fno-math-errno -fno-signed-zeros -fno-builtin -DNDEBUG -msse -msse2 -mavx -O3 -llapack -larmadillo -I "/home/georg/uni/12_sem24/ba2/matrix/blaze/"

INCLUDEDIR=../../include/

main: profile.cc
	$(CC) $(CCFLAGS) -I$(INCLUDEDIR) profile.cc
