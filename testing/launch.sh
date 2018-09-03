#!/usr/bin/env bash

n=$(($1 -1))

for experiment in $(seq 0 $n)
do
python3 testing_avgR.py name test_$experiment e 1 n 30 maze def &
done
echo "Experiments lanched"