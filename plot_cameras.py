# Visualize cameras:
from pytorch3d.vis.plotly_vis import _add_camera_trace, _add_ray_bundle_trace
from plotly.subplots import make_subplots
import os
from utils.camera_utils import (
    read_camera_parameters_world,
    normalize_cameras,
)
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras
)
import argparse

# -------------------------------------------------------------------------
#
# Arguments
#
parser = argparse.ArgumentParser()
parser.add_argument('--calibration', type=str, required=True, help='Path to calibration file.')
parser.add_argument('--scale', type=float, default=0.05, help='Scale of cameras.')
args = parser.parse_args()

Ks, Rs, ts = read_camera_parameters_world(args.calibration)
Rs, ts = normalize_cameras(Rs, ts)

if Ks != []:
    cameras_fov = FoVPerspectiveCameras(K=Ks, R=Rs[:,:3,:3], T=ts)
else: 
    cameras_fov = FoVPerspectiveCameras(R=Rs[:,:3,:3], T=ts)

fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
_add_camera_trace(
                    fig, cameras_fov, "Calibration Cameras", 0, 1, args.scale
                )

R, T = look_at_view_transform(dist=1.0, elev=0, azim=0)
camera = FoVOrthographicCameras(max_y=0.5, min_y=-0.5, max_x=0.5, min_x=-0.5, R=R, T=T)

_add_camera_trace(
                fig, camera, "Visualisation Camera", 0, 1, args.scale
            )
fig.show()