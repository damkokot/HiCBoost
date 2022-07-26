#!/usr/bin/env python3

# import models
import custom_model
import hic_model

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


def load_args(args=None):
	""" Parse command-line arguments. """
	parser = argparse.ArgumentParser(description='Create custom model for sequences extracted from HiC maps')
	parser.add_argument('-pm','--model', type=str, help='h5 file with trained model')
	parser.add_argument('-p', '--params', type=str, help='file with parameters')
	# parser.add_argument('-o', '--out_dir', type=str, help='path in which model will be stored')
	return parser.parse_args(args)


def merge(target_model, hic_model):
	"""
	Merging target model with hic model.
	Outputs from those will be passed to dense block
	that contains two Dense layers. Final output is
	probability that specific target sequence given
	its neighbors is accesibile in specific cell type.
	Cell type is then determined on the basis of hic sequences.
	"""

	# concatenate outputs from models
	outs = tf.keras.layers.concatenate([target_model.output,hic_model.output], axis=1)

	# set common denseblock for target and hic model
	dense_merge = tf.keras.layers.Dense(16,
		name='dense_merge')

	current = dense_merge(outs)

	# apply batch norm and dropout
	batch_normalization_merge = tf.keras.layers.BatchNormalization(momentum=0.90, 
		gamma_initializer=None, name='batch_normalization_merge')

	current = batch_normalization_merge(current)
	
	# apply dropout
	dropout_merge = tf.keras.layers.Dropout(rate=0.2, name='dropout_merge')

	current = dropout_merge(current)

	# set final dense
	dense_final_merge = tf.keras.layers.Dense(1,
		name='dense_finale_merge',
		activation='sigmoid')

	current = dense_final_merge(current)

	# set final model, plot, and display summary
	model = tf.keras.Model([target_model.input, hic_model.input], current)

	tf.keras.utils.plot_model(model, to_file='final_merged_model_hic.png', show_shapes=True)

	return model


def main(args=None):
	args = load_args(args)

	# load pretrained model
	pre_model = custom_model.load_trained_md(args.model)

	# load parameters
	params_model_target = custom_model.get_dense(args.params)

	# set models
	for_target_model = custom_model.build_dense(params_model_target, pre_model)

	for_hic_model = hic_model.conv_block(pre_model)

	# call model
	merge(for_target_model, for_hic_model)


if __name__ == "__main__":
	main(sys.argv[1:])
