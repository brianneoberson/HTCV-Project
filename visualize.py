import torch
import trimesh
from models.nerf import (
    NeuralRadianceField
)
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
)
from torchmcubes import marching_cubes
from utils.create_target_images import create_target_images
from utils.read_camera_parameters import read_camera_parameters

#cow_cameras, cow_images, cow_silhouettes = generate_cow_renders(num_views=40, azimuth_range=180)
target_silhouettes = create_target_images("./data/input/extracted_frames_todler")
K, R, T = read_camera_parameters("./data/input/extracted_frames_todler/calibration_160906_ian5.json") # should parse calibration file as command line arg
target_cameras = FoVPerspectiveCameras(K=K, R=R, T=T)
print(f'Number of target cameras: {len(target_cameras)}')
print(f'Generated {len(target_silhouettes)} silhouettes/cameras.')

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

checkpoint = torch.load("model_checkpoint_it3500.pt")
nerf = NeuralRadianceField()
print("Loading checkpoint...")
nerf.load_state_dict(checkpoint['model_state_dict'])

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
mesh.export("output.ply", file_type="ply")

print("Exported to PLY file.")
