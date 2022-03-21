#!/usr/bin/env python3

import sys
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

def load_h5(pred_file: str, target_file: str):
	"""
	Given hdf format files, load probability
	predictions and test targets values.

	parameter pred_file, target_file: name of the files or path to them
	return: vector of targets true labels and vector of probablity predictions
	"""
	pred_obj = h5py.File(pred_file)
	tg_obj  = h5py.File(target_file)
	pred_set = np.array(pred_obj.get('preds'))
	tg_set = np.array(tg_obj.get('targets'))

	assert tg_obj['targets'].shape == pred_obj['preds'].shape, 'Target vector and predictions vecto must be the same shape!'
	return tg_set, pred_set

if __name__ == '__main__':
	pred_path = sys.argv[1]
	target_path = sys.argv[2]
	load_h5(pred_path, target_path)
