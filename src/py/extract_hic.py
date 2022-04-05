#!/usr/bin/env python3
import argparse
import numpy as np
import hicstraw

# hic = hicstraw.HiCFile("GSE63525_K562_combined_30.hic")

# start = 74443629
# end = 74443821
# m = int(((start+end)/2)/5000)*5000
# start = m - 1000000
# end = m + 1000000
# mat_obj_chr5 = hic.getMatrixZoomData('5','5', 'observed', 'KR', 'BP', 5000 )
# numpy_mat_obj_chr5 = mat_obj_chr5.getRecordsAsMatrix(start, end, m , m + 4999 )
# print(len(numpy_mat_obj_chr5))



def load_test(bedfile):
	"""
	Loading bed format file with test sequences.
	Extract them and return a list with sequences.
	"""
	test_seqs = []
	with open(bedfile, 'r') as bf:
		bf = bf.readlines()
		for line in bf:
			line = line.strip().split('\t')
			chrom, startidx, endidx = line[:3]
			test_seq = (chrom, startidx, endidx)
			test_seqs.append(test_seq)
	yield test_seqs


def hic_sequences(test_sequences, hic_data):
	"""
	Creating vector with contact frequencies values for each
	test sequence. Given the vector extract 20 most significant
	contacts. Those contacts are then transform to sequences.
	Returning dictionary where keys are targest and values are 
	list with contacts.
	""" 



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting best neighbours of target sequences from HiC data')
	parser.add_argument('hic', type=str, help='HiC format file')
	parser.add_argument('--targets', type=str, help='bed format file with test targets')
	# parser.add_argument('--resolution', type=int, help='resolution for contact matrix')
	# parser.add_argument('--range', type=int)
	args = parser.parse_args()

	hic = hicstraw.HiCFile(args.hic)
	print(hic.getMatrixZoomData('5','5', 'observed', 'KR', 'BP', 5000))
	test_seqs = load_test(args.targets)


	



### TODO
### parse sequences test, put each sequence to getMatrixZoomData object, create numpy end-startx1
### take 20 highest scores, calculate their distance (200 bin )