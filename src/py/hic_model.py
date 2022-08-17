#!/usr/bin/env python3
import custom_model

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
	parser = argparse.ArgumentParser(description='Create custom model for sequences extracted from HiC maps')
	parser.add_argument('-pm','--model', type=str, help='h5 file with trained model')
	parser.add_argument('-d', '--dataset', type=str, help='path to data directory with tfr records and parameters')
	parser.add_argument('-p', '--params', type=str, help='file with parameters')
	# parser.add_argument('-o', '--out_dir', type=str, help='path in which model will be stored')
	return parser.parse_args(args)


def load_params(params_file):
	# read model parameters
	with open(params_file) as params_open:
		params = json.load(params_open)
		params_model = params['model']
		params_train = params['train']
	return params_model, params_train

# def build_block(current, block_params, seqnn_model):
# 	"""based on basenji seqnn.py script"""

# 	# get parameters
# 	block_args = {}
# 	block_name = block_params['name']

# 	pass_all_globals = True

# 	# set global defaults
# 	global_vars = ['activation', 'batch_norm', 'bn_momentum', 'norm_type', 
# 	'l2_scale', 'l1_scale', 'padding', 'kernel_initializer']
# 	for gv in global_vars:
# 		gv_value = getattr(seqnn_model, gv, False)
# 		if gv_value and pass_all_globals:
# 			block_args[gv] = gv_value
	
#     # set remaining params
# 	block_args.update(block_params)
# 	del block_args['name']

# 	# switch for block
# 	if block_name[0].islower():
# 		block_func = blocks.name_func[block_name]
# 		current = block_func(current, **block_args)

# 	else:
# 		block_func = blocks.keras_func[block_name]
# 		current = block_func(**block_args)(current)

# 	return current


def conv_block(trained_model, rc=True, shift=3):
	
	# FIXNE: freeze model before changing it
	trained_model.trainable = False
	
	# input for hic sequences (5 neigbours each has 5000 length)
	input_hic = tf.keras.Input(shape=(25000, 4), name='hic')
	
	# for each vector that represents each sequence from hic map
	# create convolutional block separetely 
	output_conv = []
	for i in range(0, 25000 - 1344, 384):
		inp_hic = input_hic[:, i:i + 1344, :]
		inp_hic._name = f"neighbour_{i}"

		current = inp_hic

		# augmentation
		if rc:
			current , reverse_bool = layers.StochasticReverseComplementHic()(current)
		if shift != [0]:
		 	current = layers.StochasticShiftHic(shift)(current)


		loaded_model = tf.keras.Model(trained_model.get_layer('conv1d').input, 
			trained_model.get_layer('dense_1').output)

		current = loaded_model(current)
		
		if rc:
			current = layers.SwitchReverseHic(None)([current, reverse_bool])

		# add output from each conv block to a list
		output_conv.append(current)

	
	# model for conv blocks
	conv_model = tf.keras.Model(inputs=input_hic, outputs=output_conv)

	# set dense layer, common for outputs from each conv block
	dense_common = tf.keras.layers.Dense(16, 
	activation='sigmoid', 
	name='dense_common',
	use_bias=True, 
	kernel_initializer='he_normal', 
	kernel_regularizer=tf.keras.regularizers.l1_l2(0, 0))
	
	# pass each output from conv blocks
	# to common dense layer
	out_convs = []
	for out_conv in conv_model.output:
		current = dense_common(out_conv)
		out_convs.append(current)

	# here add concatenate layer
	current = tf.keras.layers.concatenate(out_convs)

	# then pass to dense layer and get (1,164) shape output
	dense_final_hic = tf.keras.layers.Dense(164, 
		activation='sigmoid', 
		name='dense_final_hic',
		use_bias=True, 
		kernel_initializer='he_normal', 
		kernel_regularizer=tf.keras.regularizers.l1_l2(0, 0))
	current = dense_final_hic(current)

	# set model
	full_model = tf.keras.Model(inputs=input_hic, outputs=current)

	tf.keras.utils.plot_model(full_model, to_file='just_a_model_hic.png', show_shapes=True)

	return full_model

def main(args=None):
	args = load_args(args)

	loaded_model = custom_model.load_trained_md(args.model)
	# tf.keras.utils.plot_model(loaded_model, to_file='model.png', show_shapes=True)
	params_model = load_params(args.params)[0]
	# params_train = custom_model.train_params(args.params)
	conv_block(loaded_model)


if __name__ == "__main__":
	main(sys.argv[1:])