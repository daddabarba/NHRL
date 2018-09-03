#!/usr/bin/env bash

n=$(($2 -1))

for experiment in $(seq 0 $n)
do
python3 testing_avgR.py name $1_$experiment e 1 n 30 maze def &
done
echo "Experiments lanched"