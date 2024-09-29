#!/bin/bash

cd build/
#variants="-DONE_VEC -DTWO_VEC -DNESTED -DWRAPPED -DARMADILLO -DBLAZE -DLAPACK -DEIGEN"
#variants="-DONE_VEC -DTWO_VEC -DNESTED -DWRAPPED -DBLAZE -DEIGEN"
#variants="-DONE_VEC -DTWO_VEC -DNESTED -DWRAPPED -DARMADILLO -DBLAZE"
compiler="-msse -msse2 -mavx -mavx2 -O3 -DNDEBUG"

#cmake ./ -DMY_FLAGS="-DREAL_NON_SYMM_HESS -DNDEBUG ${variants} ${compiler}"
#cmake ./ -DMY_FLAGS="-DCOMPLEX_NON_SYMM_HESS -DREAL_NON_SYMM_HESS -DNDEBUG ${variants} ${compiler}"
#cmake ./ -DMY_FLAGS="-DCOMPLEX_NON_SYMM_HESS -DREAL_NON_SYMM_HESS -DOLD_PARAM -DROWWISE -DNDEBUG ${variants} ${compiler}"
cmake ./ -DMY_FLAGS="-DCOMPLEX_NON_SYMM_HESS -DREAL_NON_SYMM_HESS -DOLD_PARAM ${compiler}"
#cmake ./ -DMY_FLAGS="-DCOMPLEX_SYMM_HESS -DREAL_SYMM_HESS -DLAPACK ${compiler}"
#cmake ./ -DMY_FLAGS="-DREAL_SYMM_FULL -DCOMPLEX_SYMM_FULL -DEIGEN ${compiler}"

#cmake ./ -DMY_FLAGS="-DFULL -DLONGREAL_SYMM -DNDEBUG ${variants} ${compiler}"
#cmake ./ -DMY_FLAGS="-DFULL -DLONGREAL_SYMM -DNDEBUG ${variants} ${compiler}"
#cmake ./ -DMY_FLAGS="-DFULL -DREAL_SYMM -DNDEBUG ${variants} ${compiler}"
make -B evaluate
#for tol in "1e-16" "1e-11" "1e-14"
#for tol in "1e-12"
#do
#  echo "tol $tol"
#  for seed in {1..10}
#  do
#    echo "seed $seed"
#    echo "compiler options $compiler"
#    echo $(date +%T)
#    ./evaluation/evaluate 900 1000 $seed $tol
#  done
#done

cd ../
