#!/bin/bash
for i in $(seq 1 20);
do
  echo $i
    ./build/eigen_O1 1 500 $i
done
echo "All done"
