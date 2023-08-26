import torch
import trimesh
import os
from omegaconf import OmegaConf
import argparse
from models.nesc import NeSC
from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    AbsorptionOnlyRaymarcher,
    ImplicitRenderer,
    FoVOrthographicCameras,
    look_at_view_transform
)
from torchmcubes import marching_cubes

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
    parser.add_argument('--output', type=str, required=True, help="Path to output file (make sure to append the .ply extension).")
    parser.add_argument('--mc_thresh', type=float, default=0.005, help="Level set threshold for the marching cubes algorithm.")
    args = parser.parse_args()

    experiment_folder = os.path.join(os.path.dirname(args.chkpt), "..")
    config = OmegaConf.load(os.path.join(experiment_folder, "config.yaml"))

    print("Loading checkpoint...")
    nesc = NeSC.load_from_checkpoint(args.chkpt, config=config).to("cpu")

    # create a camera positioned at (0, 0, 1) looking in the negative z direction
    R, T = look_at_view_transform(dist=1.0, elev=0, azim=0)
    camera = FoVOrthographicCameras(max_y=0.5, min_y=-0.5, max_x=0.5, min_x=-0.5, R=R, T=T)

    raysampler_grid = NDCMultinomialRaysampler(
        image_width=128,
        image_height=128,  
        n_pts_per_ray= 128, 
        min_depth= 0.5,
        max_depth= 1.5, 
    )
    raymarcher = AbsorptionOnlyRaymarcher()
    renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher)
    raybundle = raysampler_grid(camera)

    # Export
    export_mesh(nesc, raybundle, args.output, args.mc_thresh)

    print("Done.")

if __name__ == '__main__':
    main()