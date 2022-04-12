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

def check_range(margin_point, res, rng):
	"""
	checking if provided range or resolution values
	lead indexes of sequence of interest to be outside
	of chromosome's ends
	"""
	return [margin - rng, margin + rng, margin + res]

def set_lenghts(hic_data):
	"""
	Information about each chromosome length
	"""
	chrom_len = {}
	for chrom in hic_data:
		chrom_len.setdefault(chrom.name, chrom.length)
	return chrom_len

def hic_sequences(test_sequences, hic_data, res, rng):
	"""
	Creating vector with contact frequencies values for each
	test sequence. Given the vector extract 5 most significant
	contacts. Those contacts are then transform to sequences.
	Returning dictionary where keys are targest and values are 
	list with contacts.
	""" 
	# chrom length
	chrom_len = set_lenghts(hic_data)


	# keeping objects of chromosome matrix
	# hint: make function from it
	chr_obj = []
	for record in test_sequences:
		chr_num, start_seq, end_seq = record
		seq_len = int(end_seq) - int(start_seq)
		if chr_num not in chr_obj:
			contact_obj = hic_data.getMatrixZoomData(chr_num[3:], chr_num[3:], 'observed', 'KR', 'BP', res)
			chr_obj.append(chr_num)
		# set marginal to create midpoint of the region of interest
		margin = int(((int(start_seq)+int(end_seq))/2)/res)*res
		
		# set indexes for rows and columns
		if check_range(margin, res, rng)[0] <= 0:
			start_row, end_row = 0, margin + rng - 1
			start_col, end_col = margin, margin + res - 1
		elif check_range(margin, res, rng)[1] > chrom_len[chr_num[3:]]:
			start_row, end_row = margin - rng, chrom_len[chr_num[3:]] - 1
			start_col, end_col = margin, margin + res - 1
		elif check_range(margin, res, rng)[2] > chrom_len[chr_num[3:]]:
			start_row, end_row = margin - rng, margin + rng - 1
			start_col, end_col = margin, chrom_len[chr_num[3:]] - 1
		else:
			start_row, end_row = margin - rng, margin + rng - 1
			start_col, end_col = margin, margin + res - 1
		
		# set contact vector
		contact_vector = contact_obj.getRecordsAsMatrix(start_row, end_row, start_col, end_col)
		
		# check shape of contact vector
		if contact_vector.shape != (1, 1):
		
		# get best scores of bins
		# given that set range of contact sequences
		# NOT SURE IF IT IS IMPLEMENTED RIGHT!!!
			best_contacts = np.argpartition(contact_vector, 5, axis=None)[-5:]
			for contact_bin in best_contacts:
				# set beginning and end of bin
				# split it by step of 100 bp
				for seq_point in range(begin_bin, end_bin, 100):
					# not finished
					contact_margin = int((contact_bin*res)-(res/2))
					start_ct, end_ct = int(contact_margin-seq_len/2), int(contact_margin+seq_len/2)
					print(chr_num, start_ct, end_ct)
			

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting best neighbours of target sequences from HiC data')
	parser.add_argument('hic', type=str, help='HiC format file')
	parser.add_argument('--targets', type=str, help='bed format file with test targets')
	parser.add_argument('--resolution', type=int, default=1000, help='resolution for contact matrix')
	parser.add_argument('--range', type=int, 
		help='range, to and from, the center of target sequence; value should be multiplicity of resolution')
	### ADD ARGUMENT FOR OUTPUT FILES
	args = parser.parse_args()


	hic = hicstraw.HiCFile(args.hic)
	test_seqs = load_test(args.targets)
	hic_sequences(test_seqs, hic, args.resolution, args.range)
	# contact_obj = hic.getMatrixZoomData('11', '11', 'observed', 'KR', 'BP', 5000)
	# contact_vector = contact_obj.getRecordsAsMatrix(0, 2000000-1, 130000, 2000000-1)
	# print(contact_vector)
