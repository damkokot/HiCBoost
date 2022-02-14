#!/bin/sh

# ENCODE
wget -r ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgDnaseUniform

# rearrange
mv hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgDnaseUniform/* data/basenji_peak/encode
rm -r hgdownload.cse.ucsc.edu

# Roadmap
wget -r -A "*DNase.hotspot.fdr0.01.peaks.bed.gz" http://egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/narrowPeak

# rearrange
mv egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/narrowPeak/archive_DNase_hotspot_peaks/* data/basenji_peak/roadmap
rm -r egg2.wustl.edu
