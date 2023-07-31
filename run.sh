#!/bin/bash
make O1 CPPFLAGS="-DSINGLE -DIMPLICIT -DHALF -DFULL" BASEVERSION=10
for i in $(seq 1 20);
do
  echo $i
    #./build/O1 1 500 $i > a/test_eigen_multiple_runs_$i
    ./build/O1 1 500 $i
done
echo "Implicit done"
make O1 CPPFLAGS="-DSINGLE" BASEVERSION=20
for i in $(seq 1 20);
do
  echo $i
    #./build/O1 1 500 $i > a/test_eigen_multiple_runs_$i
    ./build/O1 1 500 $i
done
echo "Single done"
make O1 BASEVERSION=30
for i in $(seq 1 20);
do
  echo $i
    #./build/O1 1 500 $i > a/test_eigen_multiple_runs_$i
    ./build/O1 1 500 $i
done
echo "Double done"
make O1 CPPFLAGS="-DEIGEN -DHALF" BASEVERSION=99
for i in $(seq 1 20);
do
  echo $i
    ./build/O1 1 500 $i > a/test_eigen_func_runs_$i
done
echo "All done"
