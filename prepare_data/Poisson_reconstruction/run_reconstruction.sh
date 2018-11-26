#!/bin/bash
# usage: ./sample_mesh /mnt/external/points/data/ModelNet40 /mnt/external/points/data/ModelNet40_PD
num_procs=16

inputDir=$1
name="$2"
myDir=$(pwd)
echo "input: $inputDir output: $outputDir extension: $name"

cd $inputDir
find . -type d -exec mkdir -p "$outputDir"/{} \;

scriptFile="$myDir/meshlab_poisson_reconstruction.mlx"
inverscript="$myDir/invert_orientation.mlx"
function meshlab_poisson_reconstruct () {
	iFile="$1"
	iName="$(basename $iFile)"
	# remove last extension
	iName="${iName%.*}"
	iDir="$(dirname $iFile)"
	oFile="$iDir/$iName"_poisson.ply
	oFile2="$iDir/$iName"_inverted_poisson.ply
	sFile="$2"
	sFile2="$3"
	meshlabserver -i $iFile -o $oFile -s $sFile
	meshlabserver -i $oFile -o $oFile2 -s $sFile2

	# if [ ! -f "$oFile" ]; then
		# meshlabserver -i $iFile -o $oFile -s $sFile
		# meshlabserver -i $oFile -o $oFile2 -s $sFile2
	# fi
}
export -f meshlab_poisson_reconstruct

echo $scriptFile
find $inputDir -type f -print
find . -type f -name "$name" | xargs -P $num_procs -I % bash -c 'meshlab_poisson_reconstruct "$@"' _ % $scriptFile $inverscript
cd $myDir
