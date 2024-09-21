#!/bin/bash

cd build/
variants="-DONE_VEC -DTWO_VEC -DNESTED -DWRAPPED -DARMADILLO -DBLAZE"
compiler="-msse -msse2 -mavx -mavx2 -O3"

cmake ./ -DMY_FLAGS="-DFULL -DREAL_SYMM -DNDEBUG ${variants} ${compiler}"
make -B evaluate
for seed in {1..10}
do
  echo "seed $seed"
  echo "compiler options $compiler"
  echo $(date +%T)
  ./evaluation/evaluate 1 1000 $seed
done

cd ../
