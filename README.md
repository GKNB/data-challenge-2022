# data-challenge-2022

First step is to do data preprocessing. This can be found in data-preprocess-ae.ipynb and data-preprocess-gan.ipynb

Second step is to train/load the autoencoder model. Model are saved in either autoencoder_\<num1\>, where \<num1\> is the number of dimension of latent space, or autoencoder_18_hier_\<num2\>, where \<num2\> here represent the number of subnetwork in a hierarchcal architechture. To see how to load them, please refer to ae-v1-backup.ipynb

Third step is to train/load the GAN model. The pretrained generator Model is saved in gan_v1, while a generator/discriminator model that is badly trained is saved in gan_v2. To see how to load the pretrained generator model, or how to load the g/d model, see
