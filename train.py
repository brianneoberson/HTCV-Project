import torch
import pytorch3d
# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
)
from utils.generate_cow_renders import generate_cow_renders
from utils.read_camera_parameters import read_camera_parameters

from helpers import (
    huber,
    sample_images_at_mc_locs,
    show_full_render,
)

from models.nerf import (
    NeuralRadianceField,
    HarmonicEmbedding,
)

from FrameExtractor.create_traget_images import create_target_images
# -------------------------------------------------------------------------
#
# Arguments
#


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

data_dir = "./Segmentation/segment-anything-main/opt"
#cow_cameras, cow_images, cow_silhouettes = generate_cow_renders(num_views=40, azimuth_range=180)
target_silhouettes = create_target_images(data_dir)
K, R, T = read_camera_parameters("./Segmentation/segment-anything-main/opt/calibration_160906_ian5.json") # should parse calibration file as command line arg
target_cameras = FoVPerspectiveCameras(K=K, R=R, T=T)
print(f'Number of target cameras: {len(target_cameras)}')
print(f'Generated {len(target_silhouettes)} images/silhouettes/cameras.')




# -------------------------------------------------------------------------
#
# Implicit Renderers
#

# render_size describes the size of both sides of the 
# rendered images in pixels. Since an advantage of 
# Neural Radiance Fields are high quality renders
# with a significant amount of details, we render
# the implicit function at double the size of 
# target images.
render_size = target_silhouettes.shape[1] * 2

# Our rendered scene is centered around (0,0,0) 
# and is enclosed inside a bounding box
# whose side is roughly equal to 3.0 (world units).
volume_extent_world = 3.0

# 1) Instantiate the raysamplers.

# Here, NDCMultinomialRaysampler generates a rectangular image
# grid of rays whose coordinates follow the PyTorch3D
# coordinate conventions.
raysampler_grid = NDCMultinomialRaysampler(
    image_height=render_size,
    image_width=render_size,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

# MonteCarloRaysampler generates a random subset 
# of `n_rays_per_image` rays emitted from the image plane.
raysampler_mc = MonteCarloRaysampler(
    min_x = -1.0,
    max_x = 1.0,
    min_y = -1.0,
    max_y = 1.0,
    n_rays_per_image=750,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

# 2) Instantiate the raymarcher.
# Here, we use the standard EmissionAbsorptionRaymarcher 
# which marches along each ray in order to render
# the ray into a single 3D color vector 
# and an opacity scalar.
raymarcher = EmissionAbsorptionRaymarcher()

# Finally, instantiate the implicit renders
# for both raysamplers.
renderer_grid = ImplicitRenderer(
    raysampler=raysampler_grid, raymarcher=raymarcher,
)
renderer_mc = ImplicitRenderer(
    raysampler=raysampler_mc, raymarcher=raymarcher,
)


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
torch.manual_seed(1)

# Instantiate the radiance field model.
neural_radiance_field = NeuralRadianceField().to(device)

# Instantiate the Adam optimizer. We set its master learning rate to 1e-3.
lr = 1e-3
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr)

# We sample 6 random cameras in a minibatch. Each camera
# emits raysampler_mc.n_pts_per_image rays.
batch_size = 6

# 3000 iterations take ~20 min on a Tesla M40 and lead to
# reasonably sharp results. However, for the best possible
# results, we recommend setting n_iter=20000.
n_iter = 100

# Init the loss history buffers.
loss_history_sil = []

# The main optimization loop.
for iteration in range(n_iter):      
    # In case we reached the last 75% of iterations,
    # decrease the learning rate of the optimizer 10-fold.
    if iteration == round(n_iter * 0.75):
        print('Decreasing LR 10-fold ...')
        optimizer = torch.optim.Adam(
            neural_radiance_field.parameters(), lr=lr * 0.1
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
    
    # The optimization loss is a simple
    # sum of the color and silhouette errors.
    loss = sil_err
    
    # Log the loss history.
    loss_history_sil.append(float(sil_err))
    
    # Every 10 iterations, print the current values of the losses.
    if iteration % 10 == 0:
        print(
            f'Iteration {iteration:05d}:' + f' loss silhouette = {float(sil_err):1.2e}'
        )

    # Take the optimization step.
    loss.backward()
    optimizer.step()

    if iteration % 20 == 0:
        # Additional information
        PATH = "model.pt"

        torch.save({
                    'epoch': iteration,
                    'model_state_dict': neural_radiance_field.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, PATH)
    
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

