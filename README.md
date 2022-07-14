# data-challenge-2022

First step is to do data preprocessing. This can be found in data-preprocess-ae.ipynb and data-preprocess-gan.ipynb. Notice that in gan preprocessing, we standardize the input, so we need to factor the output in the future.

Second step is to train/load the autoencoder model. Model are saved in either autoencoder_\<num1\>, where \<num1\> is the number of dimension of latent space, or autoencoder_18_hier_\<num2\>, where \<num2\> here represent the number of subnetwork in a hierarchcal architechture. To see how to load weights, please refer to autoencoder_72/ae-v1.ipynb, to see how I generate hierarchical autoencoder and their performance, please refer to ae-v1-backup.ipynb

Third step is to train/load the GAN model. The pretrained generator Model is saved in gan_v1, while a generator/discriminator model that is badly trained is saved in gan_v2. To see how to load the pretrained generator model, or how to load the g/d model, see


## TODO:

### Autoencoder:

Summarize all the loss

Write a function which interpret the physical meaning of latent vector, maybe a reconstruct + plot function for (latent_vec_1), (latent_vec_1 + letent_vec_2), ..., and then look at their different

Look at PCA performance as comparison

(Should we do that?) Look at POD performance as comparison

### GAN:

Write a function which can be used to reconstuct the wind velocity field and plot it

Find some good measure indicating whether the output is good

Try to use a larger alpha

Try to do multiple updates\

	https://stackoverflow.com/questions/71151303/in-a-gan-with-custom-training-loop-how-can-i-train-the-discriminator-more-times
	
Try to work with multiple GPU

	https://www.tensorflow.org/guide/distributed_training#use_tfdistributestrategy_with_custom_training_loops
	
Need to figure out if the model is training, what the loss should look like, what is a good measure of if things are going on well

	https://stackoverflow.com/questions/42690721/how-to-interpret-the-discriminators-loss-and-the-generators-loss-in-generative
	
See if there is any other work on physics based SRGAN

Tips for training

	https://github.com/soumith/ganhacks
	
	https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9
	
Maybe we should add BN layer back???

See if we can improve the performance with the following:

	https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#end-to-end_example_a_gan_training_loop_from_scratch
