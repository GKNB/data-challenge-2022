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
		#default values: learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08
		#generator optimizer
		self.g_opt = tf.keras.optimizers.Adam(learning_rate=parameters['train']['learning_rate'], 
												beta_1=parameters['train']['beta_1'], 
												beta_2=parameters['train']['beta_2'], 
												epsilon=parameters['train']['epsilon'])
		#discriminator optimizer
		if self.is_GAN == True:
			self.d_opt = tf.keras.optimizers.Adam(learning_rate=parameters['train']['learning_rate'], 
												beta_1=parameters['train']['beta_1'], 
												beta_2=parameters['train']['beta_2'], 
												epsilon=parameters['train']['epsilon'])
 
		
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

	def compute_gen_loss(self, x_HR, x_SR, d_SR):
		if self.is_GAN == True: #generator, High-Res and Super-Res, Distriminator Super-Res
			#content loss + advers loss, return tot_loss, content_loss, adver_loss
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


	def compute_disc_loss(self, d_HR, d_SR): #discriminator, High-Res and Super-Res
		return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.concat([d_HR, d_SR], axis=0),
                             labels=tf.concat([tf.ones_like(d_HR), tf.zeros_like(d_SR)], axis=0)))

	def set_working_mode(self, is_GAN):
		self.is_GAN = is_GAN
		#use keras to train simple generator is is_Gan==False
		#use train_step to train gen/disc if is_Gan==True
		return
	@tf.function
	def train_step_GAN(self, batch_LR, batch_HR):  #basic training step used by GAN
		generator = self.gen
		discriminator = self.disc
		generator_optimizer = self.g_opt
		discriminator_optimizer = self.d_opt
		alpha_advers = self.paramters['train']['alpha_advers']
		d_loss_ideal_range = [0.45, 0.65]	
		g_reflect_max = 20  
		d_reflect_max = 20
		print("Using d_loss_ideal_range:{}, g_reflect_max: {}, d_reflect_max: {}"\
								.format(d_loss_ideal_range, g_reflect_max, d_reflect_max)) 
		#Tianle's code started here
		g_count = 0
		d_loss = tf.constant(0.0)
		d_HR = discriminator(batch_HR, training=False)
		while(d_loss < tf.constant(d_loss_ideal_range[0]) and g_count < g_reflect_max):
			g_count += 1
			with tf.GradientTape() as gen_tape:
				batch_SR = generator(batch_LR, training=True)
				d_SR = discriminator(batch_SR, training=False)
				g_loss, content_loss, g_advers_loss = generator_loss(batch_HR, batch_SR, d_SR, alpha_advers=alpha_advers)
			
			grad_of_gen = gen_tape.gradient(g_loss, generator.trainable_variables)
			generator_optimizer.apply_gradients(zip(grad_of_gen, generator.trainable_variables))
			d_loss = discriminator_loss(d_HR, d_SR)
		
		d_count = 0
		d_loss = tf.constant(100.0)
		while(d_loss > tf.constant(d_loss_ideal_range[-1]) and d_count < d_reflect_max):
			d_count += 1
			with tf.GradientTape() as disc_tape:
				batch_SR = generator(batch_LR, training=False)
				d_HR = discriminator(batch_HR, training=True)
				d_SR = discriminator(batch_SR, training=True)
				d_loss = discriminator_loss(d_HR, d_SR)
			
			grad_of_disc = disc_tape.gradient(d_loss, discriminator.trainable_variables)
			discriminator_optimizer.apply_gradients(zip(grad_of_disc, discriminator.trainable_variables))
		    
		return g_loss, d_loss, g_count, d_count


	def train_GAN(self, epochs=20):	#used only with GAN
		gen_model = self.gen
		disc_model = self.disc
		batch_size = self.parameters['train']['batch_size']
		alpha_advers = self.parameters['train']['alpha_advers']
		'''
		This method trains the generator and disctiminator adversarially
		Notice the two model should be the input of this function
		output: trained generator model and discriminator model
		'''
		train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(batch_size)
		batch_count = tf.data.experimental.cardinality(train_dataset)
		
		g_opt = self.g_opt
		d_opt = self.d_opt	
		
		# Start training
		print('Training network ...')
		for epoch in range(1, epochs+1):
			print('Epoch: %d' %(epoch))
			start_time = time()
			epoch_g_loss, epoch_d_loss, N, g_count_tot, d_count_tot = 0, 0, 0, 0, 0
			
			for batch_idx, (batch_LR, batch_HR) in enumerate(train_dataset):
				N_batch = batch_LR.shape[0]
				g_loss, d_loss, g_count, d_count \
				    = self.train_step_GAN(batch_LR)  #modified by Bigeng
				
				epoch_g_loss += g_loss * N_batch
				epoch_d_loss += d_loss * N_batch
				N += N_batch
				g_count_tot += g_count
				d_count_tot += d_count
			
			epoch_g_loss = epoch_g_loss / N       
			epoch_d_loss = epoch_d_loss / N       
			
			val_SR = gen_model.predict(x_val, verbose=0)
			val_d_HR = disc_model.predict(y_val, verbose=0)
			val_d_SR = disc_model.predict(val_SR, verbose=0)
			
			val_g_loss, val_content_loss, val_advers_loss = generator_loss(y_val, val_SR, val_d_SR, alpha_advers)
			val_d_loss = discriminator_loss(val_d_HR, val_d_SR)
			
			print('Epoch generator loss = %.6f, discriminator loss = %.6f, g_count = %d, d_count = %d' %(epoch_g_loss, epoch_d_loss, g_count_tot, d_count_tot))
			print('Epoch val: g_loss = %.6f, d_loss = %.6f, content_loss = %.6f, advers_loss = %.6f' \
			      %(val_g_loss, val_d_loss, val_content_loss, val_advers_loss))
			print('Epoch took %.2f seconds\n' %(time() - start_time), flush=True)
		
		print('Done.')
		
		return gen_model, disc_model
 
