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
		
	def load_data(self):
		parameters_data = self.parameters['data']
		data_xy_dict = get_data_xy_from_h5(parameters_data['output_folder'], 
							parameters_data['file_format'], 
							parameters_data['xy_keyword_dict'], 
							parameters_data['xy_exclude_list'])
		self.data_x = np.array(data_xy_dict['x'])
		self.data_y = np.array(data_xy_dict['y'])


	def split_data(self, test_set_ratio=0.2, val_set_subratio=0.25,rdm_state=42):
		#First split data into train+validation and test set
		self.x_train, self.x_test, self.y_train, self.y_test = \
					train_test_split(self.data_x, self.data_y, 
												test_size=test_set_ratio, random_state=rdm_state)

		#Next split training again into train and validation
		self.x_train, self.x_val, self.y_train, self.y_val = \
					train_test_split(self.x_train, self.y_train, 
												test_size=val_set_subratio, random_state=rdm_state)
		print("x_train.shape: {}".format(self.x_train.shape) )
		print("x_val.shape: {}".format(self.x_val.shape) )
		print("x_test.shape: {}".format(self.x_test.shape) )

		print("y_train.shape: {}".format(self.y_train.shape) )
		print("y_val.shape: {}".format(self.y_val.shape) )
		print("y_test.shape: {}".format(self.y_test.shape) )

		print(np.max(self.x_train), np.max(self.x_val), np.max(self.x_test), np.min(self.x_train), np.min(self.x_val), np.min(self.x_test))
		print(np.max(self.y_train), np.max(self.y_val), np.max(self.y_test), np.min(self.y_train), np.min(self.y_val), np.min(self.y_test))
	
	def load_gen_model(): 
		return

	def load_disc_model():
		return

	def save_gen_model(): #only save weights
		return
	def save_disc_model(): #only save weights
		return

	def compute_gen_loss(self):
		if self.is_GAN == True:
			#content loss + advers loss, return tot_loss, content_loss, adver_loss
			x_HR = self.x_HR
			x_SR = self.x_SR
			d_SR = self.d_SR
			alpha_advers = self.paramters['train']['alpha_advers']
			content_loss = tf.reduce_mean((x_HR - x_SR)**2)
			g_advers_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_SR, labels=tf.ones_like(d_SR)))
			g_loss = content_loss + alpha_advers * g_advers_loss
			return g_loss, content_loss, g_advers_loss
		elif self.is_GAN == False:
			print("Currently not implemented!")
			#only content loss
			return
		else:
			print("Error Unknown GAN option!") 


	def compute_disc_loss(self):
		d_HR = self.d_HR
		d_SR = self.d_SR
		return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([d_HR, d_SR], axis=0),
                             labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0)))

	def set_working_mode(self, is_GAN):
		self.is_GAN = is_GAN
		#use keras to train simple generator is is_Gan==False
		#use train_step to train gen/disc if is_Gan==True
		return
	def train_step() #used only with GAN
		return

	def train():	#used only with GAN
		return

	




 
