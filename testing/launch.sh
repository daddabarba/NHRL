#!/usr/bin/env bash

n=$(($2 -1))

for experiment in $(seq 0 $n)
do
gnome-terminal --window-with-profile=Bash -- \bash -c "python3 testing_avgR.py name $1_$experiment e 1 n 30 maze def; read"
done
echo "Experiments lanched"