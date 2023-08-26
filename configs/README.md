# Description of config parameters
To run an experiment using our model, a config file containing the model and data parameters must be given to the `main.py` script. This folder contains the config files used for the experiments mentioned in our report.
- cow_no_sc_uniform:
    - data: cow
    - loss: L_silhouette
    - sampling: uniform
- cow_sc_uniform:
    - data: cow
    - loss: L_silhouette + L_space_carving
    - sampling: uniform
- cow_sc_stratified:
    - data: cow
    - loss: L_silhouette + L_space_carving
    - sampling: stratified
- dance_sc_stratified:
    - data: panoptic studio 150821_dance285
    - loss: L_silhouette + L_space_carving
    - sampling: stratified
- dance_no_sc_stratified:
    - data: panoptic studio 150821_dance285
    - loss: L_silhouette
    - sampling: stratified

Below you will find an example config with added description of certain parameters.

```
# experiment name can also be a path
experiment_name: cow/nesc/no_sc_uniform 

# name of output directory
output_dir: output 

seed: 1

dataset: 
  root_dir: ./data/cow_data_40
  img_width: 128
  img_height: 128
  # downscale factor will downscale the images for training and evaluation
  downscale_factor: 1
  # training split ratio specifies the ratio of images from the dataset to use as training images, the rest will be used for evaluation
  training_split: 0.9

model:
  name: nesc
  nb_rays_per_image: 750
  nb_samples_per_ray: 128

  # depth (distance to camera) at which to start sampling
  min_depth: 0.1

  # volument extent describes the depth at which to stop sampling
  volume_extent_world: 3

  # size of hidden layers
  n_hidden_neurons: 256
  n_hidden_layers: 7
  n_harmonic_functions: 60

  # activation function can be one of softplus, relu, gelu, gaussian
  activation: softplus

  # in the case of gaussian activation, the mean (mu) and standard deviation (sigma) need to be specified
  mu: 0
  sigma: 1

  # if set to true, stratified sampling is used, otherwise uniform sampling is used
  stratified_sampling: False

trainer:
  device: gpu
  num_devices: 1
  max_epochs: 2000
  log_every_n_steps: 100
  # log a predicted image from training set
  log_image_every_n_epochs: 100
  check_val_every_n_epoch: 100
  batch_size: 6
  lr: 0.001

  # multiplier for the silhouette (huber) loss
  lambda_sil_err: 1.

  # multiplier for the space carving loss
  lambda_sc_err: 0.

checkpoint:
  save_every_n_epochs: 2000

```