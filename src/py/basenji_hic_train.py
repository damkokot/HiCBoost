#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import time

import pandas as pd
import numpy as np
from natsort import natsorted
import tensorflow as tf
from tensorflow import keras

import custom_model
import hic_model
import merge_model
import dataset_hic
import trainer


def load_args(args=None):
	""" Parse command-line arguments. """
	parser = argparse.ArgumentParser(description='training model for sequences from DNASeq experiment and their neighbors from HiC data')
	parser.add_argument('-pm','--model', type=str, help='path to original saved model from Basenji project')
	parser.add_argument('-d', '--dataset', type=str, help='path to data directory with tfr records')
	parser.add_argument('-p', '--params', type=str, help='file with parameters')
	parser.add_argument('-t', '--target', type=str, help='file with targets')
	parser.add_argument('-ct', '--cell_type', type=str, help='name of the cell type of interest, based on HiC data')
	parser.add_argument('-k', dest='keras_fit', default=False, action='store_true',
		help='Train with Keras fit method [Default: %default]')
	parser.add_argument('--restore',dest='restore', default=False, action='store_true')
	parser.add_argument('-mf', dest='model_file', type='str', default=None)
	parser.add_argument('-o', '--out_dir', type=str, help='path in which model will be stored')
	return parser.parse_args(args)


def load_trained_md(path_to_model):
	trained_model = tf.keras.models.load_model(path_to_model)
	return trained_model


def train_params(params_file):
	with open(params_file) as params_open:
		params = json.load(params_open)
	params_train = params['train']
	return params_train


def get_cell_id(target_file, cell_type):
	"""
	Extracting id of the cell type from target's file.
	Id will be used to set specific cell from vector 
	in dataset that is passed to the model.
	ID is chosen based on name of the cell type that should
	be the same as the cell type of HiC data.
	"""
	targets_df = pd.read_csv(target_file, index_col=0, sep='\t')
	id_cell = list(targets_df['description']).index(cell_type)
	return id_cell


def load_data(data_dir, params_train, cell_id):

	# read datasets
	train_data = []
	eval_data = []

	# load train data
	train_data.append(dataset_hic.SeqDataset(data_dir, 
		split_label='train', 
		batch_size=params_train['batch_size'], 
		shuffle_buffer=params_train.get('shuffle_buffer', 128),
		hic_length=25000,
		cell_id = cell_id,
		mode='train'))

	# load eval data
	eval_data.append(dataset_hic.SeqDataset(data_dir,
	split_label='valid',
	batch_size=params_train['batch_size'],
	hic_length = 25000,
	cell_id = cell_id,
	mode='eval',
	tfr_pattern=None))

	return train_data, eval_data


def train_and_fit(params_train, train_data, eval_data, model, out_dir, keras_fit, restore):
	"""Compiling model and perform training"""

	seqnn_trainer = trainer.Trainer(params_train, train_data,
		eval_data,out_dir)

	seqnn_trainer.compile(model)

	data_dirs = 1
	
	# train model
	if keras_fit:
		seqnn_trainer.fit_keras(model)
	else:
		if data_dirs == 1:
			seqnn_trainer.fit_tape(model)
		else:
			seqnn_trainer.fit2(model)


def main(args=None):
	args = load_args(args)

	# load pretrained model
	pre_model = tf.keras.models.load_model(args.model, compile=False)

	# load parameters
	params_train = train_params(args.params)

	# get cell type id
	cell_id = get_cell_id(args.target, args.cell_type)

	# load data
	train_data, eval_data = load_data(args.dataset, params_train, cell_id)[0], load_data(args.dataset, params_train, cell_id)[1]

	# build model for HiC sequences
	for_hic_model = hic_model.build_hic_model(pre_model)

	# call model
	if args.restore:
		full_model = tf.keras.models.load_model(args.model_file, custom_objects={'StochasticReverseComplement': layers.StochasticReverseComplement(),  
	                                                                  'SwitchReverse': layers.SwitchReverse(), 'StochasticShift': layers.StochasticShift, 
	                                                                  'GELU': layers.GELU(), 'GELU_FINAL': layers.GELU_FINAL}, compile=False)
	else:
		full_model = merge_model.merge(pre_model, for_hic_model)
	
	# train and fit
	train_and_fit(params_train, train_data, eval_data, full_model, args.out_dir, args.keras_fit)


if __name__ == "__main__":
	main(sys.argv[1:])