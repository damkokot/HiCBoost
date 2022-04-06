#!/usr/bin/env python3
import argparse
import numpy as np
import hicstraw


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
	with open(bedfile, 'r') as bf:
		bf = bf.readlines()
		for line in bf:
			record = line.strip().split('\t')[:3]
			yield record


def hic_sequences(test_sequences, hic_data, res, rng):
	"""
	Creating vector with contact frequencies values for each
	test sequence. Given the vector extract 20 most significant
	contacts. Those contacts are then transform to sequences.
	Returning dictionary where keys are targest and values are 
	list with contacts.
	""" 
	
	for record in test_sequences:
		chr_num, start_seq, end_seq = record
		# print(type(chr_num[3:]), start_seq, end_seq, res)
		contact_obj = hic_data.getMatrixZoomData(chr_num[3:], chr_num[3:], 'observed', 'KR', 'BP', res)

		# set marginal to create midpoint of the region of interest
		margin = int(((int(start_seq)+int(end_seq))/2)/res)*res
		
		# create vector with contacts
		contact_vector = contact_obj.getRecordsAsMatrix(margin-rng, margin+rng, margin, margin+res-1)
		yield contact_vector
	# 	contact_vectors.append(contact_vector)
	# print(len(contact_vectors))
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting best neighbours of target sequences from HiC data')
	parser.add_argument('hic', type=str, help='HiC format file')
	parser.add_argument('--targets', type=str, help='bed format file with test targets')
	parser.add_argument('--resolution', type=int, help='resolution for contact matrix')
	parser.add_argument('--range', type=int)
	args = parser.parse_args()

	hic = hicstraw.HiCFile(args.hic)
	test_seqs = load_test(args.targets)
	a = hic_sequences(test_seqs, hic, args.resolution, args.range)
	


	



### TODO
### parse sequences test, put each sequence to getMatrixZoomData object, create numpy end-startx1
### take 20 highest scores, calculate their distance (200 bin )