#!/usr/bin/env python3
import sys
import numpy as np
import hicstraw
import collections

import basenji


def load_test(bedfile):
	"""
	Loading bed format file with test sequences.
	Extract them and return a list with sequences.
	"""
	model_seqs = []
	with open(bedfile, 'r') as bf:
		bf = bf.readlines()
		for line in bf:
			record = line.strip().split('\t')
			model_seqs.append(ModelSeq(record[0],int(record[1]),int(record[2]),None))
	return model_seqs


def check_range(margin_point, res, rng):
	"""
	checking if provided range or resolution values
	lead indexes of sequence of interest to be outside
	of chromosome's endsw
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


def contacts(test_sequence, hic_data, res, rng):
	"""
	Creating vector with contact frequencies values for each
	test sequence. Given the vector extract 5 most significant
	contacts. Those contacts are then transform to sequences.
	Returning dictionary where keys are targest and values are 
	list with contacts.
	""" 

	ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])

	# chrom length
	chrom_len = set_lenghts(hic_data)

	msq = test_sequence
	seq_len = int(msq.end) - int(msq.start)
	
	# call matrix for each chromosome
	contact_obj = hic_data.getMatrixZoomData(msq.chr[3:], msq.chr[3:], 'observed', 'KR', 'BP', res)
	
	# set marginal to create midpoint of the region of interest
	margin = int(((int(msq.start)+int(msq.end))/2)/res)*res
	
	# set indexes for rows and columns
	# make function
	if check_range(margin, res, rng)[0] <= 0:
		start_row, end_row = 0, margin + rng - 1
		start_col, end_col = margin, margin + res - 1
	elif check_range(margin, res, rng)[1] > chrom_len[msq.chr[3:]]:
		start_row, end_row = margin - rng, chrom_len[msq.chr[3:]] - 1
		start_col, end_col = margin, margin + res - 1
	elif check_range(margin, res, rng)[2] > chrom_len[msq.chr[3:]]:
		start_row, end_row = margin - rng, margin + rng - 1
		start_col, end_col = margin, chrom_len[msq.chr[3:]] - 1
	elif check_range(margin, res, rng)[1] > chrom_len[msq.chr[3:]] and check_range(margin, res, rng)[2] > chrom_len[msq.chr[3:]]:
		start_row, end_row = margin - rng, chrom_len[msq.chr[3:]] - 1
		start_col, end_col = margin, chrom_len[msq.chr[3:]] - 1
	else:
		start_row, end_row = margin - rng, margin + rng - 1
		start_col, end_col = margin, margin + res - 1
	
	# set contact vector
	contact_vector = contact_obj.getRecordsAsMatrix(start_row, end_row, start_col, end_col)
	
	# set list to store hic seqs
	hic_ngh = []
	
	# check shape of contact vector
	if contact_vector.shape != (1, 1):
		
		# get best scores of bins
		best_contacts = np.argpartition(contact_vector, -5, axis=None)[-5:]

		for contact_bin in best_contacts:
			# set beginning and end of bin
			begin_bin, end_bin = start_row + (contact_bin * res) - res , start_row + (contact_bin * res)

			hic_seq = ModelSeq(msq.chr, begin_bin, end_bin, None)
			
			# write sequence
			hic_ngh.append(hic_seq)
	
	# set sequences if there is no output for conctact vector
	elif contact_vector.shape == (1,1):
		med_bin = 201
		best_contacts = [med_bin - 2, med_bin -1, med_bin, med_bin + 1, med_bin + 2]

		for contact_bin in best_contacts:
			# set beginning and end of bin
			begin_bin, end_bin = start_row + (contact_bin * res) - res , start_row + (contact_bin * res)
			hic_seq = ModelSeq(msq.chr, begin_bin, end_bin, None)
			# write sequence
			hic_ngh.append(hic_seq)
	
	return hic_ngh