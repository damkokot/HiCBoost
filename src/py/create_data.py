#!/usr/bin/env python3
import sys
import os
import argparse
import hicstraw

# import scripts from basenji project
import basenji_data_write_mod
from basenji_data_hic import ModelSeq
from basenji import util


def load_args(args=None):
	""" Parse command-line arguments. """
	parser = argparse.ArgumentParser(description='TFRecords creation')
	parser.add_argument('-bf','--bed_file', type=str, help='bed file with sequences of the interest')
	parser.add_argument('-ff','--fasta_file', type=str, help='fasta file with referance genome')
	parser.add_argument('-hf','--hic_file', type=str, help='HiC file with HiC data')
	parser.add_argument('-cov','--cov_files', type=str, help='path to coverage files')
	parser.add_argument('-r', '--seqs_per_tfr', type=int, help='number of records per TFRecord')
	parser.add_argument('-uc', '--umap_clip', default=1, type=float, 
		help='Clip values at unmappable positions to distribution quantiles, eg 0.25. [Default: %default]')
	parser.add_argument('-ut', '--umap_tfr', default=False, action='store_true', 
		help='Save umap array into TFRecords [Default: %default]')
	parser.add_argument('-c', '--crop', type=int, help='file with parameters')
	parser.add_argument('-p', '--processes', type=int, help='file with parameters')
	parser.add_argument('-o', '--out_dir', type=str, help='path in which TFRecords will be stored')
	return parser.parse_args(args)


def get_seqs(bedfile):
	"""
	Load sequences from sequences.bed
	"""
	model_seqs = []
	with open(bedfile, 'r') as bf:
		bf = bf.readlines()
		for line in bf:
			record = line.strip().split('\t')
			model_seqs.append(ModelSeq(record[0],int(record[1]),int(record[2]),record[3]))
	return model_seqs


def main(args=None):
	# load arguments
	args = load_args(args)
	
	# load files
	fasta_file = args.fasta_file
	seqs_bed_file = args.bed_file
	hic_file = args.hic_file
	seqs_cov_dir = args.cov_files

	# load model sequences
	mseqs = get_seqs(seqs_bed_file)

	# initialize TF Records dir
	tfr_dir = '%s/tfrecords_hic' % args.out_dir
	if not os.path.isdir(tfr_dir):
		os.mkdir(tfr_dir)

	write_jobs = []

	fold_labels = ['train', 'valid', 'test']

	for fold_set in fold_labels:
		fold_set_indexes = [i for i in range(len(mseqs)) if mseqs[i].label == fold_set]
		fold_set_start = fold_set_indexes[0]
		fold_set_end = fold_set_indexes[-1] + 1

		tfr_i = 0
		tfr_start = fold_set_start
		tfr_end = min(tfr_start+args.seqs_per_tfr, fold_set_end)

		while tfr_start <= fold_set_end:
			tfr_stem = '%s/%s-%d' % (tfr_dir, fold_set, tfr_i)

			cmd = 'CUDA_VISIBLE_DEVICES=1 src/py/basenji_data_write_mod.py'
			cmd += ' -s %d' % tfr_start
			cmd += ' -e %d' % tfr_end
			cmd += ' --umap_clip %f' % args.umap_clip
			cmd += ' -x %d' % args.crop
			# if options.umap_tfr:
			# 	cmd += ' --umap_tfr'
			# if options.umap_bed is not None:
			# 	cmd += ' -u %s' % unmap_npy

			cmd += ' %s' % fasta_file
			cmd += ' %s' % seqs_bed_file
			cmd += ' %s' % hic_file
			cmd += ' %s' % seqs_cov_dir
			cmd += ' %s.tfr' % tfr_stem


			# breaks on some OS
			# cmd += ' &> %s.err' % tfr_stem
			write_jobs.append(cmd)
			
			# else:
			# 	j = slurm.Job(cmd,
			# 	name='write_%s-%d' % (fold_set, tfr_i),
			# 	out_file='%s.out' % tfr_stem,
			# 	err_file='%s.err' % tfr_stem,
			# 	queue='standard', mem=15000, time='12:0:0')
			# 	write_jobs.append(j)

			# update
			tfr_i += 1
			tfr_start += args.seqs_per_tfr
			tfr_end = min(tfr_start+args.seqs_per_tfr, fold_set_end)

	# if options.run_local:
	util.exec_par(write_jobs, args.processes, verbose=True)
	# else:
	# slurm.multi_run(write_jobs, options.processes, verbose=True,
	# launch_sleep=1, update_sleep=5)


if __name__ == "__main__":
	main(sys.argv[1:])