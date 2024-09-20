#!/bin/bash

cd build/
###compiler="-fno-trapping-math -fno-math-errno"
###compiler="-fno-trapping-math -fno-math-errno -fno-builtin -fno-signed-zeros -fno-rtti"
#compiler="-fno-trapping-math -fno-math-errno -fno-signed-zeros -fno-builtin -msse -msse2 -mavx -O3"
#compiler="-fno-trapping-math -fno-math-errno -fno-signed-zeros -msse -msse2 -mavx -O3"
variants="-DONE_VEC -DTWO_VEC -DNESTED -DWRAPPED -DARMADILLO -DBLAZE"

compiler="-O3"
cmake ./ -DMY_FLAGS="-DFULL -DREAL_SYMM -DNDEBUG ${variants} ${compiler}"
make -B evaluate
for seed in {1..10}
do
  echo "seed $seed"
  echo "compiler options $compiler"
  echo $(date +%T)
  ./evaluation/evaluate 1 1000 $seed
done


compiler="-fno-trapping-math -fno-math-errno -fno-signed-zeros -O3"
cmake ./ -DMY_FLAGS="-DFULL -DREAL_SYMM -DNDEBUG ${variants} ${compiler}"
make -B evaluate
for seed in {1..10}
do
  echo "seed $seed"
  echo "compiler options $compiler"
  echo $(date +%T)
  ./evaluation/evaluate 1 1000 $seed
done


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


compiler="-fno-trapping-math -fno-math-errno -fno-signed-zeros -msse -msse2 -mavx -mavx2 -O3"
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
#done
