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
from models import *
from copy import deepcopy



class AWWSM4_SR_GAN:
	def __init__(self, data_x, data_y, generator=generator(), discriminator=discriminator(), is_GAN=False, parameters=dict()):
		self.gen = generator
		print(self.gen.summary())
		self.disc = discriminator
		print(self.disc.summary())
		self.is_GAN = is_GAN
		self.parameters = deepcopy(parameters)
		self.data_x = np.array(data_x)
		self.data_y = np.array(data_y)
 
