import numpy as np
import pandas as pd
import xarray as xr
import h5py
from time import time

import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Input, Lambda, LeakyReLU, Add, Dense, Activation, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform, constant, TruncatedNormal
import tensorflow as tf

from models import *
from preprocess import *
from copy import deepcopy
 

class Autoencoder(Model):
	def __init__(self, encoder, decoder):
		super(Autoencoder, self).__init__() 
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
