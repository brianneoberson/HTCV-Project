import torch
import trimesh
import os
from omegaconf import OmegaConf
import argparse
import yaml
from yaml.loader import SafeLoader
from models.nerf_light import (
    Nerf
)
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
)
from torchmcubes import marching_cubes
from utils.create_target_images import create_target_images
from utils.read_camera_parameters import read_camera_parameters

parser = argparse.ArgumentParser()
parser.add_argument('--chkpt', type=str, required=True, help='Checkpoint file.')
args = parser.parse_args()

experiment_folder = os.path.join(os.path.dirname(args.chkpt), "..")

config = OmegaConf.load(os.path.join(experiment_folder, "config.yaml"))

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

#checkpoint = torch.load(args.chkpt)
nerf = Nerf.load_from_checkpoint(args.chkpt, config=config).to("cpu")
print("Loading checkpoint...")
#nerf.load_state_dict(checkpoint['model_state_dict'])

print("Getting densities...")
with torch.no_grad():
    densities, _ = nerf(raybundle)
    densities = torch.squeeze(densities)
    print(densities.size())

    print(torch.min(densities))
    verts, faces = marching_cubes(densities,0.5)
    print(verts.size(), faces.size())



mesh = trimesh.Trimesh(vertices=verts, faces=faces)

# Save as PLY file
# output into experiment folder
# experiment_name/meshes/mesh_{chkpt_name}.ply
mesh_dir = os.path.join(experiment_folder, "meshes")
if not os.path.exists(mesh_dir): 
    os.mkdir(mesh_dir)
mesh_name = f"mesh_{os.path.basename(args.chkpt)}.ply"
output_path = os.path.join(mesh_dir, mesh_name)
mesh.export(output_path, file_type="ply")

print("Exported to PLY file.")
