#!/usr/bin/env python3
import argparse
import numpy as np
import hicstraw
import collections


# set object to assign records from bed file
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])


def load_test(bedfile):
	"""
	Loading bed format file with test sequences.
	Extract them and return a list with sequences.
	"""
	with open(bedfile, 'r') as bf:
		bf = bf.readlines()
		for line in bf:
			record = line.strip().split('\t')
			yield record


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


def keep_targets(test_sequence):
	"""
	auxiliary function to yield target sequences,
	it will help to organize output files
	"""
	msq = ModelSeq(test_sequence[0], test_sequence[1], test_sequence[2], test_sequence[3])
	yield f'{msq.chr}\t{msq.start}\t{msq.end}\t{msq.label}\ttarget\n'


def contacts(test_sequence, hic_data, res, rng):
	"""
	Creating vector with contact frequencies values for each
	test sequence. Given the vector extract 5 most significant
	contacts. Those contacts are then transform to sequences.
	Returning dictionary where keys are targest and values are 
	list with contacts.
	""" 
	# chrom length
	chrom_len = set_lenghts(hic_data)

	msq = ModelSeq(test_sequence[0], test_sequence[1], test_sequence[2], test_sequence[3])
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
	
	# check shape of contact vector
	if contact_vector.shape != (1, 1):
	
		# get best scores of bins
		best_contacts = np.argpartition(contact_vector, -5, axis=None)[-5:]
		
		neighbour_rank = 1
		for contact_bin in best_contacts:
			
			# set beginning and end of bin
			begin_bin, end_bin = start_row + (contact_bin * res) - res + 50, start_row + (contact_bin * res) - 50

			# write sequence
			yield f'{msq.chr}\t{str(begin_bin)}\t{str(end_bin)}\t{msq.label}\thic\n'
			

def write_bed(target_seqs, hic_data, res, rng, bed_output):
	with open(bed_output,'w' ) as bed_out:
		seq_id = 0
		for target_seq in target_seqs:
			bed_out.write(next(keep_targets(target_seq)))
			for nbr in contacts(target_seq, hic_data, res, rng):
				bed_out.write(nbr)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting best neighbours of target sequences from HiC data')
	parser.add_argument('hic', type=str, help='HiC format file')
	parser.add_argument('--targets', type=str, help='bed format file with test targets')
	parser.add_argument('--resolution', type=int, default=1000, help='resolution for contact matrix')
	parser.add_argument('--range', type=int, 
		help='range, to and from, the center of target sequence; value should be multiplicity of resolution')
	parser.add_argument('--bed', type=str,default='hic_sequences.bed', help='bed type file name or path to it where records will be saved')
	### ADD ARGUMENT FOR OUTPUT FILES
	args = parser.parse_args()

	hic = hicstraw.HiCFile(args.hic)
	test_seqs = load_test(args.targets)
	# write_tsv(test_seqs, hic, args.resolution, args.range, False, args.tsv)
	write_bed(test_seqs, hic, args.resolution, args.range, args.bed)