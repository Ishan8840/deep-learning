# Autoencoder

used to reconstruct high-dimensional data with a neural network while compressing the data in the process to discover a more efficient representation

consists of two networks
- encoder network: translates high-dimensional data into the latent low-dimensional code
- decoder network: recovers data from the code

a problem is that if you input random latent vector into the decoder it outputs garbage, this is what a [VAE](../02-vae/README.md) solves

### Sources
[Autoencoder](https://lilianweng.github.io/posts/2018-08-12-vae/)
