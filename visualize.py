import torch
import trimesh
import os
import numpy as np
from omegaconf import OmegaConf
import argparse
from models.nesc import (
    NeSC
)
# for testing color model 
# from models.nerf import (
#     NeRF
# )
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    AbsorptionOnlyRaymarcher,
    ImplicitRenderer,
    FoVOrthographicCameras
)
from utils.helpers import get_full_render
from torchmcubes import marching_cubes
from utils.create_target_images import create_target_images
from utils.camera_utils import (
    read_camera_parameters,
    reshape_camera_matrices
)
import cv2

def export_mesh(model, raybundle, output_path, thresh):
    with torch.no_grad():
        print("Getting densities...")
        densities, _ = model(raybundle)
        densities = torch.squeeze(densities)
        print("density size: ", densities.size())
        print("densities min value: ", torch.min(densities))
        print("densities max value: ", torch.max(densities))
        print("Applying Marching Cubes...")
        verts, faces = marching_cubes(densities,thresh)
        verts = verts - torch.mean(verts)
        print("nb vert/faces: ", verts.size(), faces.size())

    print("Exporting mesh...")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(output_path, file_type="ply")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chkpt', type=str, required=True, help='Checkpoint file.')
    parser.add_argument('--mc_thresh', type=float, default=0.5, help="Level set threshold for the marching cubes algorithm.")
    parser.add_argument('--cam_id', type=int, default=0, help="Camera ID from which to generate the RayBundle for the \
                        marching cubes samples. This parameter should be irrelevant, it is only used for debugging.")
    args = parser.parse_args()

    experiment_folder = os.path.join(os.path.dirname(args.chkpt), "..")
    config = OmegaConf.load(os.path.join(experiment_folder, "config.yaml"))

    # read the camera parameters from the calibration file
    # choose first camera as the view from which to generate the RayBundles (it shouldn't matter which one we pick)
    calib_filepath = os.path.join(config.dataset.root_dir, "calibration.json")
    Ks, Rs, Ts = read_camera_parameters(calib_filepath)
    if Ks != []:
        camera = FoVOrthographicCameras(K=Ks[args.cam_id].unsqueeze(0),R=Rs[args.cam_id].unsqueeze(0), T=Ts[args.cam_id].unsqueeze(0))
    else:
        camera = FoVOrthographicCameras(R=Rs[args.cam_id].unsqueeze(0), T=Ts[args.cam_id].unsqueeze(0))

    print("Loading checkpoint...")
    nesc = NeSC.load_from_checkpoint(args.chkpt, config=config).to("cpu")

    # TODO: intialise raysample with config and/or cmd line args?
    raysampler_grid = NDCMultinomialRaysampler(
        image_width=128,
        image_height=128,  
        n_pts_per_ray= 128, 
        min_depth= 1.5, #model.config.min_depth 
        max_depth= config.model.volume_extent_world + 1, # added 1 here since otherwise looked too squashed
    )
    raymarcher = AbsorptionOnlyRaymarcher()
    renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher)
    raybundle = raysampler_grid(camera)

    # Create export directory
    mesh_dir = os.path.join(experiment_folder, "meshes")
    if not os.path.exists(mesh_dir): 
        os.mkdir(mesh_dir)
    experiment_name = config.experiment_name.replace("/","_")
    mesh_name = f"mesh_{experiment_name}_thresh={args.mc_thresh}_cam_id={args.cam_id}.ply"
    output_path = os.path.join(mesh_dir, mesh_name)

    # Export
    export_mesh(nesc, raybundle, output_path, args.mc_thresh)

    print("Done.")

if __name__ == '__main__':
    main()