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
	return [margin_point - rng, margin_point + rng, margin_point + res]

def set_lenghts(hic_data):
	"""
	Information about each chromosome length
	"""
	chrom_len = {}
	for chrom in hic_data.getChromosomes():
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
	chr_obj = {}
	seq_num = 0
	for record in test_sequences:
		chr_num, start_seq, end_seq = record
		seq_len = int(end_seq) - int(start_seq)
		if chr_num[3:] not in chr_obj:
			contact_obj = hic_data.getMatrixZoomData(chr_num[3:], chr_num[3:], 'observed', 'KR', 'BP', res)
			chr_obj[chr_num] = contact_obj
		else:
			concat_obj = chr_obj[chr_num]
		
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
		# print(contact_vector[199])
		# check shape of contact vector
		print(chr_num, start_seq, end_seq, 0, 0)
		if contact_vector.shape != (1, 1):
		
			# get best scores of bins
			best_contacts = np.argpartition(contact_vector, 5, axis=None)[-5:]
			# print(best_contacts)
			neighbour_rank = 1
			for contact_bin in best_contacts:
				
				# set beginning and end of bin
				begin_bin, end_bin = start_row + (contact_bin * res) - res + 50, start_row + (contact_bin * res) - 50
				
				# split it by step of 100 bp
				flank = 1
				for seq_point in range(begin_bin, end_bin, 500):
					start_point, end_point = int(seq_point-seq_len/2), int(seq_point+seq_len/2)
					if start_point > 0 and end_point < chrom_len[chr_num[3:]]:
						print(chr_num, start_point, end_point, neighbour_rank, flank)
						flank += 1
				neighbour_rank += 1
		seq_num += 1

	
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