# Visualize cameras:
from pytorch3d.vis.plotly_vis import _add_camera_trace
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from utils.camera_utils import (
    read_camera_parameters,
    read_camera_parameters_world,
    normalize_cameras,
    get_center_scale,
)
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PerspectiveCameras
)

data_dir = "data/cow_data_40"
cam_scale = 0.025 # 10 for dance data, 0.1 for cow data
calibration_filepath = os.path.join(data_dir, "calibration.json")
Ks, Rs, ts = read_camera_parameters_world(calibration_filepath)
Rs, ts = normalize_cameras(Rs, ts)
# center, scale = get_center_scale(ts)
#ts -= center
#ts *= scale
cameras_fov = FoVPerspectiveCameras(R=Rs, T=ts)
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
_add_camera_trace(
                    fig, cameras_fov, "fov", 0, 1, cam_scale
                )
fig.show()