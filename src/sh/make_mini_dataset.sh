#!/bin/sh

fasta=data/basenji_peak/hg19.ml.fa
blacklist=data/basenji_peak/wgEncodeHg19ConsensusSignalArtifactRegions.bed

if [ ! -e $fasta ]
then
	curl -o $fasta https://storage.googleapis.com/basenji_tutorial_data/hg19.ml.fa
	samtools faidx $fasta
fi

if [ ! -e $blacklist ]
then
	curl -O https://personal.broadinstitute.org/anshul/projects/encode/rawdata/blacklists/wgEncodeHg19ConsensusSignalArtifactRegions.bed.gz
	gunzip wgEncodeHg19ConsensusSignalArtifactRegions.bed.gz
	mv wgEncodeHg19ConsensusSignalArtifactRegions.bed data/basenji_peak
fi

basenji/bin/basenji_data.py -b $blacklist --local -p 8 -r 4096 -w 192 -l 1344 --peaks -v .12 -t .12 --stride 192 --stride_test 192 --crop 576 -o data/basenji_peak/basset_model $fasta config/targets.txt
