# Visualize cameras:
from pytorch3d.vis.plotly_vis import _add_camera_trace, _add_ray_bundle_trace
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from utils.camera_utils import (
    read_camera_parameters,
    read_camera_parameters_world,
    normalize_cameras,
    get_center_scale,
)
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    NDCMultinomialRaysampler,
    AbsorptionOnlyRaymarcher,
    ImplicitRenderer,
    FoVOrthographicCameras
)
import torch

data_dir = "data/150821_dance285"
cam_scale = 0.025 # 10 for dance data, 0.1 for cow data
calibration_filepath = os.path.join(data_dir, "calibration.json")
Ks, Rs, ts = read_camera_parameters_world(calibration_filepath)
Rs, ts = normalize_cameras(Rs, ts)
# center, scale = get_center_scale(ts)
# ts -= center
# ts *= scale
cameras_fov = FoVPerspectiveCameras(K=Ks, R=Rs[:,:3,:3].transpose(1,2), T=ts)
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
_add_camera_trace(
                    fig, cameras_fov, "fov", 0, 1, cam_scale
                )
R, T = look_at_view_transform(dist=1.0, elev=0, azim=0)
camera = FoVOrthographicCameras(max_y=0.5, min_y=-0.5, max_x=0.5, min_x=-0.5, R=R, T=T)
cam_scale = 1

# TODO: intialise raysample with config and/or cmd line args?
raysampler_grid = NDCMultinomialRaysampler(
    image_width=128,
    image_height=128,  
    n_pts_per_ray= 128, 
    min_depth= 0.5, #model.config.min_depth 
    max_depth= 1.5, 
)
raymarcher = AbsorptionOnlyRaymarcher()
renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher)
raybundle = raysampler_grid(camera)
_add_camera_trace(
                fig, camera, "fov", 0, 1, cam_scale
            )
_add_ray_bundle_trace(
                fig=fig,
                ray_bundle=raybundle,
                trace_name="raybundle",
                subplot_idx=0,
                ncols=1,
                max_rays=10,
                max_points_per_ray=128,
                marker_size=3,
                line_width=3,
            )
fig.show()