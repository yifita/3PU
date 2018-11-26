#!/bin/bash
# usage: ./run_pd_sampling /mnt/external/points/data/ModelNet40 /mnt/external/points/data/ModelNet40_PD "*.ply" 10000
num_procs=16

input_dir=$1
output_dir=$2
ext="$3"
numSample="$4"
myDir=$(pwd)
binary=$(myDir)/PdSampling
echo "input: $input_dir output: $output_dir extension: $ext"

cd $input_dir
find . -type d -exec mkdir -p $output_dir/{} \;

function poisson_disk () {
	iFile="$1"
	iName="$(basename $iFile)"
	# remove last extension
	iName="${iName%.*}"
	iDir="$(dirname $iFile)"
	oDir="$2/$iDir"
	oFile="$oDir/$iName.xyz"
	numSample="$3"
	binary="$4"
	echo "$iName"
	if [ ! -f "$oFile" ]; then
		$binary $numSample $iFile $oFile
	fi
}
export -f poisson_disk

echo $scriptFile
find . -type f -wholename "$ext" | xargs -P $num_procs -I % bash -c 'poisson_disk "$@"' _ % $output_dir $numSample $binary
cd $myDir
