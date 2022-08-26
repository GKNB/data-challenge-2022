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



# Autoencoder models
## Basic elements of encoder and decoder
def encoder(latent_dim, input_shape = (192, 192, 2)):
	'''
	return an encoder which encodes the input image into a latent vector with dimension latent_dim
	'''
	
	X_input = Input(input_shape)
	
	#FIXME Should we add BN layer? I currently add that between conv and relu for the first 4 sets of layers
	X = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(X_input)
	X = BatchNormalization()(X)
	X = LeakyReLU(alpha=0.2)(X)
	X = MaxPooling2D(pool_size=(2, 2), padding="same")(X)
	
	X = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding="same")(X)
	X = BatchNormalization()(X)
	X = LeakyReLU(alpha=0.2)(X)
	X = MaxPooling2D(pool_size=(2, 2), padding="same")(X)
	
	X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(X)
	X = BatchNormalization()(X)
	X = LeakyReLU(alpha=0.2)(X)
	X = MaxPooling2D(pool_size=(2, 2), padding="same")(X)
	
	X = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same")(X)
	X = BatchNormalization()(X)
	X = LeakyReLU(alpha=0.2)(X)
	X = MaxPooling2D(pool_size=(2, 2), padding="same")(X)
	
	#FIXME Should we add some dropout layer to regularize the model? 
	#I didn't do that, but need to look at train/val error
	
	X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	X = MaxPooling2D(pool_size=(2, 2), padding="same")(X)
	
	X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	X = MaxPooling2D(pool_size=(2, 2), padding="same")(X)
	
	X = Flatten()(X)
	X = Dense(units=latent_dim)(X)
	#FIXME Should we add an activation layer here? I didn't do it
	
	model = Model(inputs = X_input, outputs = X)
	return model


def decoder(latent_dim):
	'''
	return an encoder which encodes the input image into a latent vector with dimension latent_dim
	'''
	
	X_input = Input((latent_dim))
	
	X = Dense(units=3*3*64, input_dim=latent_dim)(X_input)
	X = Reshape((3,3,64))(X)
	
	X = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding="same")(X)
	X = BatchNormalization()(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	X = Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding="same")(X)
	X = BatchNormalization()(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	X = Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), padding="same")(X)
	X = BatchNormalization()(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	X = Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), padding="same")(X)
	X = BatchNormalization()(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	X = Conv2DTranspose(filters=16, kernel_size=(3,3), strides=(2,2), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	X = Conv2DTranspose(filters=16, kernel_size=(3,3), strides=(2,2), padding="same")(X)
	X = LeakyReLU(alpha=0.2)(X)
	
	X = Conv2D(filters=2, kernel_size=(3,3), strides=(1,1), padding="same")(X)    
	
	model = Model(inputs = X_input, outputs = X)
	return model

##hiearchical AE's
class Autoencoder_hier_1(Model):
	def __init__(self, encoder, decoder):
		super(Autoencoder_hier_1, self).__init__() 
		self.encoder = encoder
		self.decoder = decoder
	
	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded 


class Autoencoder_hier_2(Model):
	def __init__(self, encoder, decoder, ae):
		super(Autoencoder_hier_2, self).__init__() 
		self.encoder = encoder
		self.decoder = decoder
		self.ae = ae
		self.ae.trainable = False
	
	def call(self, x):
		encoded = self.encoder(x)
		latent_1 = self.ae.encoder(x)
		latent_all = Concatenate(axis=-1)([encoded, latent_1])
		decoded = self.decoder(latent_all)
		return decoded

class Autoencoder_hier_3(Model):
	def __init__(self, encoder, decoder, ae):
		super(Autoencoder_hier_3, self).__init__() 
		self.encoder = encoder
		self.decoder = decoder
		self.ae = ae
		self.ae.trainable = False
	
	def call(self, x):
		encoded = self.encoder(x)
		latent_1 = self.ae.ae.encoder(x)
		latent_2 = self.ae.encoder(x)
		latent_all = Concatenate(axis=-1)([encoded, latent_1, latent_2])
		decoded = self.decoder(latent_all)
		return decoded

class Autoencoder_hier_4(Model):
	def __init__(self, encoder, decoder, ae):
		super(Autoencoder_hier_4, self).__init__() 
		self.encoder = encoder
		self.decoder = decoder
		self.ae = ae
		self.ae.trainable = False
	
	def call(self, x):
		encoded = self.encoder(x)
		latent_1 = self.ae.ae.ae.encoder(x)
		latent_2 = self.ae.ae.encoder(x)
		latent_3 = self.ae.encoder(x)
		latent_all = Concatenate(axis=-1)([encoded, latent_1, latent_2, latent_3])
		decoded = self.decoder(latent_all)
		return decoded
