import numpy as np
import pandas as pd
import xarray as xr
import h5py

import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input, Lambda, LeakyReLU, Add, Dense, Activation, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform, constant, TruncatedNormal

import tensorflow as tf



def generator(input_shape = (96, 96, 2), nf = 64, r = 2):
	"""
	Arguments:
	input_shape -- shape of the images of the dataset, H*W*C
	nf -- integer, the number of filters in all convT layer before super-resolution step
	r -- integer, resolution ratio between output and input
	
	Returns:
	model -- a Model() instance in Keras
	"""
	
	C0 = input_shape[2]
	# Define the input as a tensor with shape input_shape
	X_input = Input(input_shape)
	
	# Define kernel size and stride used
	k, stride = 3, 1
	
	# Shall we use a mirror padding and finally cutoff the edge, like the paper does? FIXME
	X = Conv2DTranspose(filters=nf, kernel_size=(k, k), strides=(stride, stride), padding='same')(X_input)
	# Shall we use relu, or leaky_relu? FIXME
	X = Activation('relu')(X)
	
	skip_connection = X
	
	for i in range(16):
		X_shortcut = X
		
		X = Conv2DTranspose(filters=nf, kernel_size=(k, k), strides=(stride, stride), padding='same')(X)
		X = Activation('relu')(X)
		X = Conv2DTranspose(filters=nf, kernel_size=(k, k), strides=(stride, stride), padding='same')(X)
		X = Add()([X, X_shortcut])
		# Are we missing a relu activation here, if we follow the resnet paper? FIXME
	
	X = Conv2DTranspose(filters=nf, kernel_size=(k, k), strides=(stride, stride), padding='same')(X)
	X = Add()([X, skip_connection])
	
	# Start to perform sr
	nf_sr = (r**2) * nf
	X = Conv2DTranspose(filters=nf_sr, kernel_size=(k, k), strides=(stride, stride), padding='same')(X)
	
	sub_layer = Lambda(lambda x:tf.nn.depth_to_space(x,r))
	X = sub_layer(X)
	X = Activation('relu')(X)
	
	X = Conv2DTranspose(filters=C0, kernel_size=(k, k), strides=(stride, stride), padding='same')(X)
	
	model = Model(inputs = X_input, outputs = X)
	
	return model


def discriminator(input_shape = (192, 192, 2)):
	"""
	Arguments:
	input_shape -- shape of the images of the dataset, H*W*C
	
	Returns:
	model -- a Model() instance in Keras
	"""
	
	C0 = input_shape[2]
	# Define the input as a tensor with shape input_shape
	X_input = Input(input_shape)
	
	#conv1
	X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(X_input)
	X = LeakyReLU(alpha=0.2)(X)
	
	#conv2
	X = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#conv3
	X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#conv4
	X = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#conv5
	X = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#conv6
	X = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#conv7
	X = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#conv8
	X = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	X = Flatten()(X)
	
	#first fully-connect
	k_init = TruncatedNormal(stddev=0.02)
	X = Dense(units=1024, kernel_initializer=k_init)(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	#second fully-connect, no activation FIXME
	X = Dense(units=1, kernel_initializer=k_init)(X)
	
	model = Model(inputs = X_input, outputs = X)
	return model




def pretrain(data_x, data_y, parameters):
	#First split data into train+validation and test set
	x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
	
	#Next split training again into train and validation
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
	
	print(x_train.shape)
	print(x_val.shape)
	print(x_test.shape)
	
	print(y_train.shape)
	print(y_val.shape)
	print(y_test.shape)
	
	print(np.max(x_train), np.max(x_val), np.max(x_test), np.min(x_train), np.min(x_val), np.min(x_test))
	print(np.max(y_train), np.max(y_val), np.max(y_test), np.min(y_train), np.min(y_val), np.min(y_test))	
	#create generator model
	generator_model = generator(input_shape = (96, 96, 2))
	print(generator_model.summary())
	
	#First do pretrain generator for better convergence

	adam = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	generator_model.compile(optimizer=adam, loss=losses.MeanSquaredError())
	
	logdir = parameters["output_folder"] + datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
	
	checkpoint_filepath = parameters["checkpoint_filepath"]
	model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,\
																	save_weights_only=True,\
																	save_freq=10*40)
	training_history = generator_model.fit(x_train, y_train,
											batch_size=128,\
											epochs=1000,\
											shuffle=True,\
											validation_data=(x_val, y_val),\
											callbacks=[tensorboard_callback,\
											 model_checkpoint_callback])
	generator_model.evaluate(x_test, y_test)			













