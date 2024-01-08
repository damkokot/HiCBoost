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
	parser.add_argument('-o', '--out_dir', type=str, help='path in which model will be stored')
	return parser.parse_args(args)


def load_params(params_file):
	# read model parameters
	with open(params_file) as params_open:
		params = json.load(params_open)
		params_model = params['model']
		params_train = params['train']
	return params_model, params_train


def build_hic_model(trained_model, rc=True, shift=3):
	
	# FIXNE: freeze model before changing it
	trained_model.trainable = False
	
	# input for hic sequences (5 neigbours each has 5000 length)
	# FIX: avoid hardcoding of below values
	input_hic = tf.keras.Input(shape=(25000, 4), name='hic')
	
	# for each vector that represents each sequence from hic map
	# create convolutional block separetely 
	# FIX: avoid hardcoding of below values
	currents = []
	for i in range(0, 25000 - 1344, 384):

		inp_hic = input_hic[:, i:i + 1344, :]
		inp_hic._name = f"neighbour_{i}"

		current = inp_hic

		current = trained_model(current)

		currents.append(current)
	
	# concatenate outputs from conv block
	current = tf.keras.layers.concatenate(currents)

	# set dense layer, common for outputs from each conv block
	dense_common = tf.keras.layers.Dense(16, 
	activation='sigmoid', 
	name='dense_common',
	use_bias=True, 
	kernel_initializer='he_normal', 
	kernel_regularizer=tf.keras.regularizers.l1_l2(0, 0))

	current = dense_common(current)
	
	# apply batch
	batch_normalization_hic_dense = tf.keras.layers.BatchNormalization(momentum=0.90, 
		gamma_initializer=None, name='batch_normalization_hic_dense')
	
	current = batch_normalization_hic_dense(current)

	# apply dropout
	dropout_hic = tf.keras.layers.Dropout(rate=0.2, name='dropout_hic')

	current = dropout_hic(current)

	# apply activation
	current = layers.activate(current, 'relu')

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
	print(full_model.summary())
	return full_model


def main(args=None):
	args = load_args(args)

	trained_model = tf.keras.models.load_model(args.model, compile=False)
	params_model = load_params(args.params)[0]
	build_hic_model(trained_model)


if __name__ == "__main__":
	main(sys.argv[1:])