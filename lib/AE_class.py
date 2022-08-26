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
 

class AWWSM4_HIER_AE:
	def __init__(self, parameters=dict()):
		self.list_of_AE = list()	
		self.n_sub_net = parameters['train']['n_sub_net']	
		self.latent_dim_en = parameters['train']['latent_dim_en'] 
		self.latent_dim_de_origin = parameters['train']['latent_dim_de_origin']
		self.log_format = parameters['train']['log_format']

	def	generate_AE_one_by_one(self):
		for i_AE in range(self.n_sub_net):
			n_AE = i_AE + 1	
			latent_dim_en_i = self.latent_dim_en
			print("Keeping latent dimension size for encoder same: %d" % latent_dim_en_i)
			latent_dim_de_i = self.latent_dim_de_origin * n_AE
			print("Latent dimension size for decoder of the %dth AE is: %d = %d * %d") \
					% (n_AE, latent_dim_de_i, self.latent_dim_de_origin, n_AE)
		
			#create encoder	
			encoder_name = "encoder_%d_sub%d" % (latent_dim_en_i, n_AE)
			create_encoder_cmd = "%s = encoder(%d)" % \
									(encoder_name, latent_dim_en_i)	
			print("Executing: "+create_encoder_cmd)
			exec(create_encoder_cmd)
			print_encoder_cmd = "print(%s.summary())" % encoder_name	
			print("Executing: "+print_encoder_cmd)
			exec(print_encoder_cmd)
	
			#create decoder
			decoder_name = "decoder_%d_sub%d" % (latent_dim_en_i, n_AE)
			create_decoder_cmd = "%s = decoder(%d)" % \
									(decoder_name, latent_dim_de_i)
			print("Executing: "+create_decoder_cmd)
			exec(create_decoder_cmd)

			print_decoder_cmd = "print(%s.summary())" % decoder_name
			print("Executing: "+print_decoder_cmd)
			exec(print_decoder_cmd)

			#create Autoencoder_hier_X
			HIER_AE_name_now = "autoencoder_%d_hier_%d" % (latent_dim_en_i, n_AE)
			if i_AE == 0:
				init_HIER_AE_cmd = "Autoencoder_hier_%d(%s, %s)" % (encoder_name, decoder_name) 
			elif i_AE > 0:
				init_HIER_AE_cmd = "Autoencoder_hier_%d(%s, %s, %s)" % \
												(encoder_name, decoder_name, HIER_AE_name_prev)
			
			create_HIER_AE_cmd = "%s = %s" % (HIER_AE_name_now, init_HIER_AE_cmd)
			print("Executing: " + create_HIER_AE_cmd)	
			exec(create_HIER_AE_cmd)
			compile_HIER_AE_cmd = "%s.compile(optimizer='adam', loss=losses.MeanSquaredError())" \
										% HIER_AE_name_now
			print("Executing: " + compile_HIER_AE_cmd)	
			exec(comple_HIER_AE_cmd)
			#steps to do
			logdir = 'autoencoder_%d_hier_%d/' % (latent_dim_en_i, n_AE) + datetime.now().strftime('%Y%m%d-%H%M%S')
			tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

			checkpoint_filepath = 'autoencoder_%d_hier_%d/ckp/'  % (latent_dim_en_i, n_AE)

			model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                               save_weights_only=True,
                                                               save_freq=10*40)
			#do fit
			training_history_cmd = "training_history_hier_%d" % n_AE
			fit_cmd = "autoencoder_%d_hier_%d.fit" % (latent_dim_en_i, n_AE)
			arguments_cmd_p1 = "(self.X_train, self.X_train,"
			arguments_cmd_p2 = "batch_size={}, epochs={}, shuffle={},".format(self.batch_size, 
																		self.epochs, self.shuffle)
			arguments_cmd_p3 = "validation_data=(self.X_val, self.X_val)"
			arguments_cmd_p4 = "callbacks=[tensorboard_callback, model_checkpoint_callback])"
            
			training_history_cmd_full = "%s = %s" 
						%(training_history_cmd, fit_cmd+arguments_cmd_p1+arguments_cmd_p2\
													+arguments_cmd_p3+arguments_cmd_p4) 
			print("Executing: " + training_history_cmd_full)
			exec(training_history_cmd_full)

			#test section
			if i_AE > 0:
				test_prev_cmd = "%s.evaluate(self.X_test, self.X_test)" % HIER_AE_name_prev
				print("Executing: " + test_prev_cmd)
				exec(test_prev_cmd)	
				
			test_now_cmd = "%s.evaluate(self.X_test, self.X_test)" % HIER_AE_name_now
			print("Executing: " + test_now_cmd)
			exec(test_now_cmd)	
			#end of this round: update the prev HIER
			print("The %dth round of AE done! Update the prev HIER to be the current HIER!" % n_AE)
			HIER_AE_name_prev = HIER_AE_name_now
			

