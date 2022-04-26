#!/usr/bin/env python3

import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def load_h5(pred_file: str, target_file: str):
	"""
	Given hdf format files, load probability
	predictions and test targets values.

	parameter pred_file, target_file: name of the files or path to them
	return: vector of targets true labels and vector of probablity predictions
	"""
	# Open h5 files and transform it to numpy array
	pred_obj = h5py.File(pred_file)
	tg_obj  = h5py.File(target_file)
	pred_set = np.array(pred_obj.get('preds'))
	tg_set = np.array(tg_obj.get('targets'))


	# Convert 3D array to 2D and transponse matrices
	pred_set, tg_set = pred_set.transpose(2,0,1).reshape(164,-1), tg_set.transpose(2,0,1).reshape(164,-1)
	tg_set[tg_set>0]=1

	assert tg_obj['targets'].shape == pred_obj['preds'].shape, 'Target vector and predictions vector must be the same shape!'
	return pred_set, tg_set.astype(int)
def evaluate(y_pred, y_test):
	"""
	Evaluate AUROC and AUPRC metrics from predictions and
	test labels.
	"""
	aurocs = []
	auprcs = []
	for row in range(len(y_test)):
		# calculate AUROC score	
		fpr, tpr, theshold = roc_curve(y_test[row], y_pred[row])
		auroc_score = np.round(roc_auc_score(y_test[row], y_pred[row]), 5)
		aurocs.append(auroc_score)

		# calculate AUPRC score
		precision, recall, thresholds = precision_recall_curve(y_test[row], y_pred[row])
		average_precision = np.round(average_precision_score(y_test[row], y_pred[row]), 5)
		auprcs.append(average_precision)
	return aurocs, auprcs

def create_tsv(auroc_score, auprc_score, output_path):
	"""
	Save score metrics to a tab separated values file,
	where first column presents indexes of cell types, 
	second and third column shows auroc score and 
	auprc score respectively.
	"""
	indexes = [index for index in range(len(auroc_score))]
	metrics = pd.DataFrame({'index': indexes,
				'auroc': auroc_score,
				'auprc': auprc_score})

	metrics.to_csv(f'{output_path}', sep="\t")


if __name__ == '__main__':
	pred_path = sys.argv[1]
	target_path = sys.argv[2]
	output_path = sys.argv[3] # path where tsv file should be saved
	y_pred, y_test = load_h5(pred_path, target_path)
	auroc_score, auprc_score = evaluate(y_pred, y_test)
	create_tsv(auroc_score, auprc_score, output_path)
	