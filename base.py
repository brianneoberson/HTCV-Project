import torch
# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.transforms import so3_exp_map
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher
)

from pytorch3d.renderer import ImplicitRenderer
from utils.generate_cow_renders import generate_cow_renders
from utils.read_camera_parameters import read_camera_parameters

from utils.helpers import (
    huber,
    sample_images_at_mc_locs
)
import torch.optim as optim
from models.nerf import (
    NeuralRadianceField
)
import argparse
import yaml
from yaml.loader import SafeLoader
import os

import subprocess
import pytorch_lightning as pl

from utils.create_target_images import create_target_images


class EncoderModule(pl.LightningModule):
    def __init__(self, dataset: str, experiment_name: str, name: str, nb_rays_per_image: int, 
                 nb_samples_per_ray: int, min_depth: float, volume_extent_world: int, 
                 render_size: int, batch_size: int, lr: float):
        super().__init__()

        self.root_dir=dataset
        self.experiment_name=experiment_name
        self.batch_size=batch_size
        self.lr=lr

        self.target_silhouettes = create_target_images(dataset)

        K, R, T = read_camera_parameters(os.path.join(dataset, 'calibration.json'))
        self.target_cameras = FoVPerspectiveCameras(K=K, R=R, T=T)

        print(f'Number of target cameras: {len(self.target_cameras)}')
        print(f'Loaded {len(self.target_silhouettes)} silhouettes/cameras.')

        # Here, NDCMultinomialRaysampler generates a rectangular image
        # grid of rays whose coordinates follow the PyTorch3D
        # coordinate conventions.
        raysampler_grid = NDCMultinomialRaysampler(
            image_height=render_size,
            image_width=render_size,
            n_pts_per_ray=nb_samples_per_ray,
            min_depth=min_depth,
            max_depth=volume_extent_world,
        )

        # MonteCarloRaysampler generates a random subset 
        # of `n_rays_per_image` rays emitted from the image plane.
        raysampler_mc = MonteCarloRaysampler(
            min_x = -1.0,
            max_x = 1.0,
            min_y = -1.0,
            max_y = 1.0,
            n_rays_per_image=nb_rays_per_image,
            n_pts_per_ray=nb_samples_per_ray,
            min_depth=min_depth,
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
        renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher)
        self.renderer_grid= renderer_grid
        renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher)
        self.renderer_mc=renderer_mc

        # Instantiate the radiance field model.
        self.neural_radiance_field = NeuralRadianceField()


    
     ## main
    def forward(self, x):
        # not needed
        ...
    

    def training_step(self, batch, batch_idx, optimizer_idx):
        print("training step!!!")
        #TODO check optimizer is updating correctly
        # if self.current_step == round(self.max_steps * 0.75):
        #     print('Decreasing LR 10-fold ...')
        #     optimizer = torch.optim.Adam(
        #         self.neural_radiance_field.parameters(), lr=self.trainer.lr * 0.1
        # )
            
        # Sample random batch indices.
        batch_idx_ = torch.randperm(len(self.target_cameras))[:self.batch_size]
        
        # Sample the minibatch of cameras.
        batch_cameras = FoVPerspectiveCameras(
            R = self.target_cameras.R[batch_idx_], 
            T = self.target_cameras.T[batch_idx_], 
            znear = self.target_cameras.znear[batch_idx_],
            zfar = self.target_cameras.zfar[batch_idx_],
            aspect_ratio = self.target_cameras.aspect_ratio[batch_idx_],
            fov = self.target_cameras.fov[batch_idx_],
            device = self.device,
        )

        # Evaluate the nerf model.
        rendered_silhouettes, sampled_rays = self.renderer_mc(
            cameras=batch_cameras, 
            volumetric_function=self.neural_radiance_field
        )
        
        # Compute the silhouette error as the mean huber
        # loss between the predicted masks and the
        # sampled target silhouettes.
        silhouettes_at_rays = sample_images_at_mc_locs(
            self.target_silhouettes[batch_idx_, ..., None], 
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
        
        # The optimization loss is a simple sum of the color and silhouette errors.
        loss = sil_err + consistency_loss

        print()
        print("type loss")
        print(type(loss))
        self.log('train_loss', loss)
        
        
        # TO-DO: overwrite last checkpoint if current one is better

        return loss


    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.neural_radiance_field.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(self.root_dir, batch_size=32, shuffle=True)
        return train_dataloader