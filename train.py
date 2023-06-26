import torch
# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer
)
from utils.generate_cow_renders import generate_cow_renders
from utils.read_camera_parameters import read_camera_parameters

from utils.helpers import (
    huber,
    sample_images_at_mc_locs
)

from models.nerf import (
    NeuralRadianceField
)
import argparse
import yaml
from yaml.loader import SafeLoader
import os

import subprocess


from utils.create_target_images import create_target_images
# -------------------------------------------------------------------------
#
# Arguments
#
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Config file containing all hyperparameters.')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=SafeLoader)

dataset = config['dataset']
model = config['model']
trainer = config['trainer']
checkpoint = config['checkpoint']
# add git hash to config
config['githash'] = subprocess.check_output(["git", "describe", "--always"]).strip()


# -------------------------------------------------------------------------
#
# Device
#

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    print(
        'Please note that NeRF is a resource-demanding method.'
        + ' Running this notebook on CPU will be extremely slow.'
        + ' We recommend running the example on a GPU'
        + ' with at least 10 GB of memory.'
    )
    device = torch.device("cpu")

# -------------------------------------------------------------------------
#
# Data 
#

#cow_cameras, cow_images, cow_silhouettes = generate_cow_renders(num_views=40, azimuth_range=180)
root_dir = dataset['root_dir']

<<<<<<< HEAD
print(target_cameras.R.dtype)
# -------------------------------------------------------------------------
#
# Implicit Renderers
#
=======
target_silhouettes = create_target_images(root_dir)
K, R, T = read_camera_parameters(os.path.join(root_dir, 'calibration.json'))
target_cameras = FoVPerspectiveCameras(K=K, R=R, T=T)
>>>>>>> nerf

print(f'Number of target cameras: {len(target_cameras)}')
print(f'Loaded {len(target_silhouettes)} silhouettes/cameras.')



# -------------------------------------------------------------------------
#
# Implicit Renderers (instintiate raysampler and raymarchers)
#

# Here, NDCMultinomialRaysampler generates a rectangular image
# grid of rays whose coordinates follow the PyTorch3D
# coordinate conventions.
raysampler_grid = NDCMultinomialRaysampler(
    image_height=model['render_size'],
    image_width=model['render_size'],
    n_pts_per_ray=model['nb_samples_per_ray'],
    min_depth=model['min_depth'],
    max_depth=model['volume_extent_world'],
)

# MonteCarloRaysampler generates a random subset 
# of `n_rays_per_image` rays emitted from the image plane.
raysampler_mc = MonteCarloRaysampler(
    min_x = -1.0,
    max_x = 1.0,
    min_y = -1.0,
    max_y = 1.0,
    n_rays_per_image=model['nb_rays_per_image'],
    n_pts_per_ray=model['nb_samples_per_ray'],
    min_depth=model['min_depth'],
    max_depth=model['volume_extent_world'],
)

# 2) Instantiate the raymarcher.
# Here, we use the standard EmissionAbsorptionRaymarcher 
# which marches along each ray in order to render
# the ray into a single 3D color vector 
# and an opacity scalar.
raymarcher = EmissionAbsorptionRaymarcher()

# Finally, instantiate the implicit renders
# for both raysamplers.
renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher)
renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher)


# -------------------------------------------------------------------------
#
# Train
#

# First move all relevant variables to the correct device.
renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)
target_cameras = target_cameras.to(device)
target_silhouettes = target_silhouettes.to(device)

# Set the seed for reproducibility
torch.manual_seed(config['seed'])

# Instantiate the radiance field model.
neural_radiance_field = NeuralRadianceField().to(device)

# Instantiate the Adam optimizer.
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=trainer['lr'])

# We sample 'batch_size' random cameras in a minibatch. Each camera
# emits raysampler_mc.n_pts_per_image rays.
<<<<<<< HEAD
batch_size = 6

# 3000 iterations take ~20 min on a Tesla M40 and lead to
# reasonably sharp results. However, for the best possible
# results, we recommend setting n_iter=20000.
#n_iter = 3000
n_iter = 1
=======
batch_size = trainer['batch_size']
n_iter = trainer['max_steps']
>>>>>>> nerf

# Init the loss history buffers.
loss_history_sil = []

# -------------------------------------------------------------------------
#
# Create directory for saving results
#
output_dir = os.path.join("output", config['experiment_name'])
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

# save config file
file=open(os.path.join(output_dir, 'config.yaml'),"w")
yaml.dump(config,file)
file.close()

# The main optimization loop.
for iteration in range(n_iter):      
    # In case we reached the last 75% of iterations,
    # decrease the learning rate of the optimizer 10-fold.
    if iteration == round(n_iter * 0.75):
        print('Decreasing LR 10-fold ...')
        optimizer = torch.optim.Adam(
            neural_radiance_field.parameters(), lr=trainer.lr * 0.1
        )
    
    # Zero the optimizer gradient.
    optimizer.zero_grad()
    
    # Sample random batch indices.
    batch_idx = torch.randperm(len(target_cameras))[:batch_size]
    
    # Sample the minibatch of cameras.
    batch_cameras = FoVPerspectiveCameras(
        R = target_cameras.R[batch_idx], 
        T = target_cameras.T[batch_idx], 
        znear = target_cameras.znear[batch_idx],
        zfar = target_cameras.zfar[batch_idx],
        aspect_ratio = target_cameras.aspect_ratio[batch_idx],
        fov = target_cameras.fov[batch_idx],
        device = device,
    )
    
   
    # Evaluate the nerf model.
    rendered_silhouettes, sampled_rays = renderer_mc(
        cameras=batch_cameras, 
        volumetric_function=neural_radiance_field
    )
    
    # Compute the silhouette error as the mean huber
    # loss between the predicted masks and the
    # sampled target silhouettes.
    silhouettes_at_rays = sample_images_at_mc_locs(
        target_silhouettes[batch_idx, ..., None], 
        sampled_rays.xys
    )
    
    sil_err = huber(
        rendered_silhouettes, 
        silhouettes_at_rays,
    ).abs().mean()

    consistency_loss = huber(
        rendered_silhouettes.sum(axis=0), 
        silhouettes_at_rays.sum(axis=0),
    ).abs().mean()
    
    # The optimization loss is a simple
    # sum of the color and silhouette errors.
    loss = sil_err + consistency_loss
    
    # Log the loss history.
    loss_history_sil.append(float(sil_err))
    
    # Every 10 iterations, print the current values of the losses.
    if iteration % trainer['log_every_n_steps'] == 0:
        print(
            f'Iteration {iteration:05d}:' + f' loss silhouette = {float(sil_err):1.2e}'
        )

    # Take the optimization step.
    loss.backward()
    optimizer.step()

    # TO-DO: overwrite last checkpoint if current one is better
    if iteration % checkpoint['save_every_n_steps'] == 0:
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        ckpt_path = os.path.join(checkpoint_dir, f"step-{iteration:09d}.ckpt")
        if not os.path.exists(checkpoint_dir): 
            os.mkdir(checkpoint_dir)

        torch.save({
                    'epoch': iteration,
                    'model_state_dict': neural_radiance_field.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, ckpt_path)
    # possibly delete old checkpoints
    if checkpoint['save_only_latest_checkpoint']:
        # delete everything else in the checkpoint folder
        for f in checkpoint_dir.glob("*"):
            if f != ckpt_path:
                f.unlink()
    # Visualize the full renders every 100 iterations.
    # if iteration % 100 == 0:
    #     show_idx = torch.randperm(len(target_cameras))[:1]
        # show_full_render(
        #     neural_radiance_field,
        #     FoVPerspectiveCameras(
        #         R = target_cameras.R[show_idx], 
        #         T = target_cameras.T[show_idx], 
        #         znear = target_cameras.znear[show_idx],
        #         zfar = target_cameras.zfar[show_idx],
        #         aspect_ratio = target_cameras.aspect_ratio[show_idx],
        #         fov = target_cameras.fov[show_idx],
        #         device = device,
        #     ), 
        #     target_images[show_idx][0],
        #     target_silhouettes[show_idx][0],
        #     loss_history_color,
        #     loss_history_sil,
        # )


