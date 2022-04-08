#!/usr/bin/env python3
import argparse
import numpy as np
import hicstraw


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
		seq_len = int(end_seq) - int(start_seq)
		# print(type(chr_num[3:]), start_seq, end_seq, res)
		contact_obj = hic_data.getMatrixZoomData(chr_num[3:], chr_num[3:], 'observed', 'KR', 'BP', res)

		# set marginal to create midpoint of the region of interest
		margin = int(((int(start_seq)+int(end_seq))/2)/res)*res
		
		# create vector with contacts
		# if range is too big, start index of matrix set to 0
		# and end index to multiplicity of range value
		if (margin - rng) <= 0:	
			contact_vector = contact_obj.getRecordsAsMatrix(0, 2*rng-1, margin, margin+res-1)
		else:
			contact_vector = contact_obj.getRecordsAsMatrix(margin-rng, margin+rng-1, margin, margin+res-1)
		
		# check shape of contact vector
		# should be (n,2) where n is range divided by resolution
		if contact_vector.shape == (400, 1):

		# 	# print(chr_num[3:], contact_vector)
		# 	print(chr_num, margin-rng, margin+rng-1, margin, margin+res, contact_vector.shape)
		# get best scores of bins
		# given that set range of contact sequences
		# NOT SURE IF IT IS IMPLEMENTED RIGHT!!!
			best_contacts = np.argpartition(contact_vector, 20, axis=None)[-20:]
			for contact_bin in best_contacts:
				contact_margin = int((contact_bin*res)-(res/2))
				start_ct, end_ct = int(contact_margin-seq_len/2), int(contact_margin+seq_len/2)
				print(chr_num, start_ct, end_ct, seq_len)
			

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting best neighbours of target sequences from HiC data')
	parser.add_argument('hic', type=str, help='HiC format file')
	parser.add_argument('--targets', type=str, help='bed format file with test targets')
	parser.add_argument('--resolution', type=int, help='resolution for contact matrix')
	parser.add_argument('--range', type=int, 
		help='range, to and from, the center of target sequence; value should be multiplicity of resolution')
	args = parser.parse_args()

	hic = hicstraw.HiCFile(args.hic)
	test_seqs = load_test(args.targets)
	hic_sequences(test_seqs, hic, args.resolution, args.range)
	# contact_obj = hic.getMatrixZoomData('11', '11', 'observed', 'KR', 'BP', 5000)
	# contact_vector = contact_obj.getRecordsAsMatrix(0, 2000000-1, 130000, 2000000-1)
	# print(contact_vector)
