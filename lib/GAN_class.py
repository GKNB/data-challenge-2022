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
from preprocess import *
from copy import deepcopy
 

class AWWSM4_SR_GAN:
	def __init__(self, generator=generator(), discriminator=discriminator(), is_GAN=False, parameters=dict()):
		self.gen = generator
		print(self.gen.summary())
		self.disc = discriminator
		print(self.disc.summary())
		self.is_GAN = is_GAN
		self.parameters = deepcopy(parameters)
		
	def load_data():
		parameters_data = self.parameters['data']
		data_xy_dict = get_data_xy_from_h5(parameters_data['output_folder'], 
							parameters_data['file_format'], 
							parameters_data['xy_keyword_dict'], 
							parameters_data['xy_exclude_list'])
		self.data_x = np.array(data_xy_dict['x'])
		self.data_y = np.array(data_xy_dict['y'])

	
	def load_gen_model(): 
		return

	def load_disc_model():
		return

	def save_gen_model(): #only save weights
		return
	def save_disc_model(): #only save weights
		return

	def compute_gen_loss():
		if self.is_GAN == True:
			#content loss + advers loss, return tot_loss, content_loss, adver_loss
			return
		elif self.is_GAN == False:
			#only content loss
			return
		else:
			print("Error Unknown GAN option!") 


	def compute_disc_loss():
		return

	def set_working_mode(self, is_GAN):
		self.is_GAN = is_GAN
		#use keras to train simple generator is is_Gan==False
		#use train_step to train gen/disc if is_Gan==True
		return
	def train_step() #used only with GAN
		return

	def train():	#used only with GAN
		return

	




 
