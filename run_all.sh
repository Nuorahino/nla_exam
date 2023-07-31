#!/bin/bash
for i in $(seq 1 20);
do
  echo $i
    #./build/O1 1 500 $i > a/test_eigen_multiple_runs_$i
    ./build/all_O1 1 500 $i
done
echo "Implicit done"
