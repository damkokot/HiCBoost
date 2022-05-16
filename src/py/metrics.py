#!/usr/bin/env python3

import sys
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
	pred_set[np.isnan(pred_set)] = 0
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
		# fpr, tpr, theshold = roc_curve(y_test[row], y_pred[row])
		auroc_score = np.round(roc_auc_score(y_test[row], y_pred[row]), 5)
		aurocs.append(auroc_score)

		# calculate AUPRC score
		# precision, recall, thresholds = precision_recall_curve(y_test[row], y_pred[row])
		average_precision = np.round(average_precision_score(y_test[row], y_pred[row]), 5)
		auprcs.append(average_precision)
	return aurocs, auprcs


def plot_roc(y_pred, y_test, cell_type, output_path_plot):
	fpr, tpr, theshold = roc_curve(y_test[int(cell_type)], y_pred[int(cell_type)])
	auroc_score = np.round(roc_auc_score(y_test[int(cell_type)], y_pred[int(cell_type)]), 5)


	title = output_path_plot.split('/')[-1].split('.')[0]
	plt.plot(fpr, tpr, label='AUC='+str(auroc_score))
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.title(f'{title}')
	plt.legend(loc=4)
	plt.savefig(output_path_plot)


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
	cell_id = sys.argv[3]
	output_path = sys.argv[4] # path where tsv file should be saved
	output_path_plot = sys.argv[5] # path to save plot
	y_pred, y_test = load_h5(pred_path, target_path)
	auroc_score, auprc_score = evaluate(y_pred, y_test)
	plot_roc(y_pred, y_test, cell_id, output_path_plot)
	create_tsv(auroc_score, auprc_score, output_path)

	