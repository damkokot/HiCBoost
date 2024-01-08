#!/usr/bin/env python3

### Given tsv file with hic information
### and prediction from hic, reduce
### dimensionality of the result to be within
### proper size thus it can be used in calculating
### AUROC and AUPRC metrics


import sys
import h5py
import numpy as np
import pandas as pd
import argparse




def avg_prediction_score(pred_array, start_block, end_block):
	"""
	Avaraging score from given range of block
	"""	
	pred_array = pred_array[start_block + 1: end_block]
	print(pred_array)
	sum_of_elements = 0
	if len(pred_array) > 1:	
		pred_array_tp = pred_array.transpose(2,0,1).reshape(164,-1)
		print(f'transpose:{pred_array_tp}')
		for arr in pred_array:
			sum_of_elements += arr
		return sum_of_elements/len(pred_array)
	elif len(pred_array) == 1:
		return np.zeros((1, pred_array[0].shape[2]))
	

def pred_file(tsv_file, h5_hic, out_pred):
	"""
	Save prediction file with avarage scores
	"""
	pred_obj = h5py.File(h5_hic)
	pred_array = pred_obj.get('preds')
	tsv_df = pd.read_csv(tsv_file, sep='\t')
	seq_count = sum(tsv_df['seq_id'] != '-')

	# Parse dataframe to get blocks of targets and their
	# neighbours and initialize output h5 file
	with h5py.File(out_pred, 'w') as out_h5:
		out_h5.create_dataset('preds', shape=(seq_count, pred_array.shape[1], pred_array.shape[2]), dtype='float16')

	start_block = int(tsv_df['seq_id'][0])
	seq_num = 0 
	for seq_id in range(1, len(tsv_df['seq_id'])):
		if tsv_df['seq_id'][seq_id] != '-' and int(tsv_df['seq_id'][seq_id]) != start_block:
			end_block = seq_id
			out_h5['preds'][int(tsv_df['seq_id'][start_block])] = avg_prediction_score(pred_array, start_block, end_block)
			avg_prediction_score(pred_array, start_block, end_block)
			start_block = end_block
			seq_num += 1
		elif seq_id + 1 == len(tsv_df['seq_id']):
			end_block = seq_id + 1 # exception for last row from tsv file
			out_h5['preds'][int(tsv_df['seq_id'][start_block])] = avg_prediction_score(pred_array, start_block, end_block)
			avg_prediction_score(pred_array, start_block, end_block)
			seq_num += 1
		print(f'Saved sequence number {seq_num}')
	print('Output file created.')
				

			



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Reduce prediction from HiC data by avaraging probabilities')
	parser.add_argument('-p','--pred', type=str, help='h5 file with predictions from HiC')
	parser.add_argument('--tsv', type=str,help='path to tsv file with scores from HiC and sequences extracted from it')
	parser.add_argument('--cell-id', nargs='+', 
	help='cell type id or list of ids taken from the test output')
	parser.add_argument('-o', '--output', type=str, default='avg_predict.h5')
	args = parser.parse_args()

	pred_file(args.tsv, args.pred, args.output)


