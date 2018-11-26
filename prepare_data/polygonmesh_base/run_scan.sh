#!/bin/bash
# usage: ./run_scan.sh input_dir output_dir 1 "**/*.ply"
num_procs=16

input_dir=$1
output_dir=$2
numScans="$3"
ext="$4"
myDir=$(pwd)
binary=$(myDir)/polygonmesh
numModels=$(find . -type f -wholename "$ext" | wc -l)
mkdir -p "$output_dir"
echo "input: $input_dir output: $output_dir extension: $ext"

for i in $(seq $numModels); do
	$binary $input_dir $output_dir 0 $numScans $i
done
