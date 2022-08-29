# data-challenge-2022
This is a library for SMC data challenge 2022, Challenge 3. The functions can be found in folder ```./lib```:   
	*```preprocess.py``` contains the functions for data IO and pre-processing.
	*```models.py``` contains the model definitions used in hierachical autoencoder (HIER-AE) and generative adversarial network (GAN).
	*```GAN_class.py``` contains the definition and functions used for GAN training.
	*```AE_class.py``` contains the definition and functions used for AE training.
Examples of the usage of these functions can be found in ```Example.ipynb```. Below is a brief introduction of the structure of our calculations:

First step is to do data preprocessing. Notice that in GAN preprocessing, we standardize the input, so we need to factor the output in the future.

Second step is to train/load the autoencoder model. Model are saved in either autoencoder_\<num1\>, where \<num1\> is the number of dimension of latent space, or autoencoder_18_hier_\<num2\>, where \<num2\> here represent the number of subnetwork in a hierarchcal architechture. To see how to load weights and how we generate hierarchical autoencoder and create their performance data, please refer to Section 2 of Example.ipynb.

Third step is to train/load the GAN model. The pretrained generator Model is saved in gan_v1, while a generator/discriminator model that is badly trained is saved in gan_v2. To see how to load the pretrained generator model, or how to load the g/d model, see simple-start-gan-from-github-v1.ipynb. Notice since the model and checkpoint is too large, they are not included in this repository. 


## Tips for training

### Autoencoder related:

	- Summarize the losses for all different architechture

	- Write a function which interpret the physical meaning of latent vector, maybe a reconstruct + plot function for (latent_vec_1), (latent_vec_1 + letent_vec_2), ..., and then look at their different

	- Look at PCA performance as comparison

	- Look at POD performance as comparison

### GAN related:

Write a function which can be used to reconstuct the wind velocity field and plot it

Find some good measure indicating whether the output is good
	
	- Try to work with multiple GPU

	<https://www.tensorflow.org/guide/distributed_training#use_tfdistributestrategy_with_custom_training_loops>
	
	- Set indicators for checking if the model is training: if the loss looks normal. Find what is a good measure of if things are going on well:

	<https://stackoverflow.com/questions/42690721/how-to-interpret-the-discriminators-loss-and-the-generators-loss-in-generative>
	

### Others:

	- <https://github.com/soumith/ganhacks>
	
	- <https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9>
	
	- Using Batch Normalization (BN) layer

	- Check if we can improve the performance with the following:

	<https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#end-to-end_example_a_gan_training_loop_from_scratch>
