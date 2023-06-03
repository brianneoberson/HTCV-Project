import torch
from models.nerf import (
    NeuralRadianceField,
    HarmonicEmbedding,
)

from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    MultinomialRaysampler,
)

from torchmcubes import marching_cubes

from FrameExtractor.create_traget_images import create_target_images
from utils.read_camera_parameters import read_camera_parameters

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
    image_width=128,
    image_height=128,  
    n_pts_per_ray= 128, 
    min_depth= -128, 
    max_depth= 128,
)

print("Sampling rays...")
R = torch.eye(3,3)
R = R[None, :]
t = torch.tensor([0.,0.,-2.])
t = t[None, :]
camera = FoVPerspectiveCameras(R=R, T=t)
raybundle = raysampler_grid(camera)

checkpoint = torch.load("model.pt")
nerf = NeuralRadianceField()
print("Loading checkpoint...")
nerf.load_state_dict(checkpoint['model_state_dict'])

print("Getting densities...")
densities, _ = nerf(raybundle)
densities = torch.squeeze(densities)
print(densities.size())

verts, faces = marching_cubes(densities,1.1921e-07)
print(verts.size(), faces.size())

import trimesh

mesh = trimesh.Trimesh(vertices=verts, faces=faces)

# Save as PLY file
mesh.export("output.ply", file_type="ply")

print("Exported to PLY file.")
