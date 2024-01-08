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


def keep_targets(test_sequence, seq_id, is_bed=False):
	"""
	auxiliary function to yield target sequences,
	it will help to organize output files
	"""
	chr_num, start_seq, end_seq = test_sequence
	if not is_bed:
		yield f'{seq_id}\t{chr_num}\t{start_seq}\t{end_seq}\t0\t0\n'
	else:
		yield f'{chr_num}\t{start_seq}\t{end_seq}\ttest\n'


def contacts(test_sequence, hic_data, res, rng, is_bed=False):
	"""
	Creating vector with contact frequencies values for each
	test sequence. Given the vector extract 5 most significant
	contacts. Those contacts are then transform to sequences.
	Returning dictionary where keys are targest and values are 
	list with contacts.
	""" 
	# chrom length
	chrom_len = set_lenghts(hic_data)

	chr_num, start_seq, end_seq = test_sequence
	seq_len = int(end_seq) - int(start_seq)
	
	# call matrix for each chromosome
	contact_obj = hic_data.getMatrixZoomData(chr_num[3:], chr_num[3:], 'observed', 'KR', 'BP', res)
	
	# set marginal to create midpoint of the region of interest
	margin = int(((int(start_seq)+int(end_seq))/2)/res)*res
	
	# set indexes for rows and columns
	# make function
	if check_range(margin, res, rng)[0] <= 0:
		start_row, end_row = 0, margin + rng - 1
		start_col, end_col = margin, margin + res - 1
	elif check_range(margin, res, rng)[1] > chrom_len[chr_num[3:]]:
		start_row, end_row = margin - rng, chrom_len[chr_num[3:]] - 1
		start_col, end_col = margin, margin + res - 1
	elif check_range(margin, res, rng)[2] > chrom_len[chr_num[3:]]:
		start_row, end_row = margin - rng, margin + rng - 1
		start_col, end_col = margin, chrom_len[chr_num[3:]] - 1
	elif check_range(margin, res, rng)[1] > chrom_len[chr_num[3:]] and check_range(margin, res, rng)[2] > chrom_len[chr_num[3:]]:
		start_row, end_row = margin - rng, chrom_len[chr_num[3:]] - 1
		start_col, end_col = margin, chrom_len[chr_num[3:]] - 1
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
			
			# split it by step of 100 bp
			flank = 1
			for seq_point in range(begin_bin, end_bin+1, 100):
				start_point, end_point = int(seq_point-seq_len/2), int(seq_point+seq_len/2)
				if start_point > 0 and end_point < chrom_len[chr_num[3:]]:
					if not is_bed:
						yield f'-\t{chr_num}\t{str(start_point)}\t{str(end_point)}\t{str(neighbour_rank)}\t{str(flank)}\t{float(contact_vector[contact_bin])}\n'
					elif is_bed:
						yield f'{chr_num}\t{str(start_point)}\t{str(end_point)}\ttest\n'
					flank += 1
			neighbour_rank += 1


def write_tsv(target_seqs, hic_data, res, rng, is_bed, tsv_output):
	with open(tsv_output, 'w') as tsv_out:
		tsv_out.write('seq_id\tchrom_id\tstart\tend\tneighbour_rank\tflank\tcontact_score\n')
		seq_id = 0
		for target_seq in target_seqs:
			tsv_out.write(next(keep_targets(target_seq,seq_id,is_bed)))
			for nbr in contacts(target_seq, hic_data, res, rng, is_bed):
				tsv_out.write(nbr)
			seq_id += 1


def write_bed(target_seqs, hic_data, res, rng, is_bed, bed_output):
	with open(bed_output,'w' ) as bed_out:
		seq_id = 0
		for target_seq in target_seqs:
			bed_out.write(next(keep_targets(target_seq,seq_id,is_bed)))
			for nbr in contacts(target_seq, hic_data, res, rng, is_bed):
				bed_out.write(nbr)		


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting best neighbours of target sequences from HiC data')
	parser.add_argument('hic', type=str, help='HiC format file')
	parser.add_argument('--targets', type=str, help='bed format file with test targets')
	parser.add_argument('--resolution', type=int, default=1000, help='resolution for contact matrix')
	parser.add_argument('--range', type=int, 
		help='range, to and from, the center of target sequence; value should be multiplicity of resolution')
	parser.add_argument('--tsv', type=str, default='hic_out.tsv',help='file name or path where records will be saved in tsv format')
	parser.add_argument('--bed', type=str,default='hic_out.bed', help='bed type file name or path to it where records will be saved')
	### ADD ARGUMENT FOR OUTPUT FILES
	args = parser.parse_args()

	hic = hicstraw.HiCFile(args.hic)
	test_seqs = load_test(args.targets)
	write_tsv(test_seqs, hic, args.resolution, args.range, False, args.tsv)
	# write_bed(test_seqs, hic, args.resolution, args.range, True, args.bed)