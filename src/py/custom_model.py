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

# load basenji tools
import dataset
import layers
import seqnn
import blocks
import trainer


def load_args(args=None):
	""" Parse command-line arguments. """
	parser = argparse.ArgumentParser(description='Create custom model for target sequence')
	parser.add_argument('-pm','--model', type=str, help='h5 file with trained model')
	parser.add_argument('-d', '--dataset', type=str, help='path to data directory with tfr records and parameters')
	parser.add_argument('-p', '--params', type=str, help='file with parameters')
	parser.add_argument('-o', '--out_dir', type=str, help='path in which model will be stored')
	return parser.parse_args(args)


def load_trained_md(path_to_model):
	trained_model = tf.keras.models.load_model(path_to_model)
	return trained_model


def reverse(trained_model):
	layer, reverse_bool = trained_model.get_layer('stochastic_reverse_complement').output
	return reverse_bool


def load_data(data_dir, params_file):
	# read model parameters
	with open(params_file) as params_open:
		params = json.load(params_open)
	params_model = params['model']
	params_train = params['train']

	# read datasets
	train_data = []
	eval_data = []

	# load train data
	train_data.append(dataset.SeqDataset(data_dir, 
		split_label='train', batch_size=params_train['batch_size'], 
		shuffle_buffer=params_train.get('shuffle_buffer', 128),  mode='train'))

	# load eval data
	eval_data.append(dataset.SeqDataset(data_dir,
	split_label='valid',
	batch_size=params_train['batch_size'],
	mode='eval',
	tfr_pattern=None))

	return train_data, eval_data


def get_dense(params_file):
	with open(params_file) as params_open:
		params = json.load(params_open)
	params_model = params['model']

	dense_params = []
	for d in params_model['trunk']:
		if 'dense' in d['name']:
			dense_params.append(d)
	params_model['trunk'] = dense_params

	return params_model


def train_params(params_file):
	with open(params_file) as params_open:
		params = json.load(params_open)
	params_train = params['train']
	return params_train


def build_dense(params, trained_model):
	seqnn_model = seqnn.SeqNN(params,
		trained_model.get_layer('batch_normalization_7').output)

	current = trained_model.get_layer('batch_normalization_7').output
	
	# get parameters
	block_params = seqnn_model.trunk[0]
	block_args = {}
	block_name = block_params['name']

	pass_all_globals = True

	# set global defaults
	global_vars = ['activation', 'batch_norm', 'bn_momentum', 'norm_type', 
	'l2_scale', 'l1_scale', 'padding', 'kernel_initializer']
	for gv in global_vars:
		gv_value = getattr(seqnn_model, gv, False)
		if gv_value and pass_all_globals:
			block_args[gv] = gv_value
	
    # set remaining params
	block_args.update(block_params)
	del block_args['name']

	# build trunk dense
	current = layers.activate(current, 'relu')

	if block_args['flatten']:
		_, seq_len, seq_depth = current.shape
		current = tf.keras.layers.Reshape((1,seq_len*seq_depth,))(current)

	# dense
	current = tf.keras.layers.Dense(
	units=block_args['units'],
	use_bias=(block_args['norm_type'] is None),
	kernel_initializer='he_normal',
	kernel_regularizer=tf.keras.regularizers.l1_l2(0, 0)
	)(current)

	# normalize
	norm_type = block_args['norm_type']

	if norm_type == 'batch-sync':
		current = tf.keras.layers.experimental.SyncBatchNormalization(
		momentum=block_args['bn_momentum'], gamma_initializer=norm_gamma)(current)
	elif norm_type == 'batch':
		current = tf.keras.layers.BatchNormalization(name='batch_dense',
		momentum=block_args['bn_momentum'], gamma_initializer=None)(current)
	elif norm_type == 'layer':
		current = tf.keras.layers.LayerNormalization(
		gamma_initializer=None)(current)

	# droput
	if block_args['dropout'] > 0:
		current = tf.keras.layers.Dropout(rate=block_args['dropout'])(current)

	# trunk model
	trunk_output = current
	trunk_model = tf.keras.Model(inputs=trained_model.input, outputs=trunk_output)
	
	# build head dense
	current = layers.activate(current, 'relu')
	
	head_params = seqnn_model.head[0]
	
	current = seqnn_model.build_block(current, head_params)

	reverse_bool = reverse(trained_model)
	current = layers.SwitchReverse(None)([current, reverse_bool])
	# trunk_output = current

	model = tf.keras.Model(inputs=trained_model.input, outputs=current)
	return model


def train_and_fit(params_train, train_data, eval_data, model, out_dir, keras_fit='True'):
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
	
	print('Loading model..')
	trained_model = load_trained_md(args.model)

	print('Setting training and validation data..')
	train_data, eval_data = load_data(args.dataset, args.params)[0], load_data(args.dataset, args.params)[1]  

	print('Getting dense layers of head and layers of trunk from the architecture...')
	params_model = get_dense(args.params)
	params_train = train_params(args.params)
	cst_model = build_dense(params_model, trained_model)
	cst_model.summary()
	# train_and_fit(params_train, train_data, eval_data, cst_model, args.out_dir)
	

if __name__ == "__main__":
	main(sys.argv[1:])
