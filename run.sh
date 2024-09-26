#!/bin/bash

cd build/
variants="-DONE_VEC -DTWO_VEC -DNESTED -DWRAPPED -DARMADILLO -DBLAZE -DLAPACK -DEIGEN"
#variants="-DONE_VEC -DTWO_VEC -DNESTED -DWRAPPED -DBLAZE -DEIGEN"
#variants="-DONE_VEC -DTWO_VEC -DNESTED -DWRAPPED -DARMADILLO"
compiler="-msse -msse2 -mavx -mavx2 -O3"

cmake ./ -DMY_FLAGS="-DFULL -DREAL_SYMM -DNDEBUG -DUSELAPACK ${variants} ${compiler}"
#cmake ./ -DMY_FLAGS="-DFULL -DLONGREAL_SYMM -DFLOAT_SYMM -DNDEBUG -DUSELAPACK ${variants} ${compiler}"
#cmake ./ -DMY_FLAGS="-DFULL -DLONGREAL_SYMM -DNDEBUG ${variants} ${compiler}"
#cmake ./ -DMY_FLAGS="-DFULL -DREAL_SYMM -DNDEBUG ${variants} ${compiler}"
make -B evaluate
for tol in "1e-8" "1e-11" "1e-14"
do
  echo "tol $tol"
  for seed in {1..10}
  do
    echo "seed $seed"
    echo "compiler options $compiler"
    echo $(date +%T)
    ./evaluation/evaluate 1 1000 $seed $tol
  done
done

cd ../
