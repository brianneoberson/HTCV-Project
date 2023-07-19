import torch
import trimesh
import os
import numpy as np
from omegaconf import OmegaConf
import argparse
from models.nerf_light import (
    Nerf
)
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    AbsorptionOnlyRaymarcher,
    ImplicitRenderer
)
from utils.helpers import get_full_render
from torchmcubes import marching_cubes
from utils.create_target_images import create_target_images
from utils.read_camera_parameters import read_camera_parameters
import cv2



def export_mesh(model, raybundle, output_path):
    print("Exporting mesh...")
    with torch.no_grad():
        densities, _ = model(raybundle)
        densities = torch.squeeze(densities)
        print("density size: ", densities.size())
        print("densities min value: ", torch.min(densities))
        print("densities max value: ", torch.max(densities))
        verts, faces = marching_cubes(densities,0.5)
        print("nb vert/faces: ", verts.size(), faces.size())

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(output_path, file_type="ply")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt', type=str, required=True, help='Checkpoint file.')
    args = parser.parse_args()

    experiment_folder = os.path.join(os.path.dirname(args.chkpt), "..")
    config = OmegaConf.load(os.path.join(experiment_folder, "config.yaml"))

    print("Loading checkpoint...")
    nerf = Nerf.load_from_checkpoint(args.chkpt, config=config).to("cpu")

    # these camera parameters were copied from the first camera in the dance calibration.json
    K = np.asarray([
        [1397.21,0,952.422,0],
        [0,1393.36,560.555,0],
        [0,0,1,0],
        [0,0,0,1]
    ])
    R = np.asarray([
        [0.05364196748,0.02402159371,0.9982712569],
        [0.6374152614,0.768712766,-0.05274910412],
        [-0.7686509766,0.6391428999,0.02592353476]
    ])
    t = np.asarray([
        [6.644493687],
        [105.724495],
        [361.4694299]
    ]).reshape(-1, 3)
    K = K[None,:]
    R = R[None, :]
    camera = FoVPerspectiveCameras(K=K, R=R, T=t)

    raysampler_grid = NDCMultinomialRaysampler(
        image_width=128,
        image_height=128,  
        n_pts_per_ray= 128, 
        min_depth= 0, 
        max_depth= 128,
    )
    raymarcher = AbsorptionOnlyRaymarcher()
    renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher)
    raybundle = raysampler_grid(camera)

    # Render a view from the camera
    render_silhouette = get_full_render(model=nerf, camera=camera, renderer=renderer_grid)
    cv2.imwrite(os.path.join(experiment_folder, "silhouette.png"), render_silhouette)
    
    # Create export directory
    mesh_dir = os.path.join(experiment_folder, "meshes")
    if not os.path.exists(mesh_dir): 
        os.mkdir(mesh_dir)
    mesh_name = f"mesh_{os.path.basename(args.chkpt)}.ply"
    output_path = os.path.join(mesh_dir, mesh_name)

    # Export
    export_mesh(nerf, raybundle, output_path)

    print("Done.")

if __name__ == '__main__':
    main()