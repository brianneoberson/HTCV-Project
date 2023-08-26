import pytorch_lightning as pl
import os
from utils.camera_utils import (
    read_camera_parameters,
    get_center_scale,
    local_to_world,
    world_to_local
)
from utils.create_target_images import create_target_images
from pytorch3d.renderer import (
    FoVPerspectiveCameras
)
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import glob
import json
from PIL import Image, ImageOps
import cv2

class SilhouetteDataset(Dataset):
    def __init__(self, config) -> None:
        super().__init__()
        self.data_dir = os.path.join(config.dataset.root_dir, "silhouettes")
        f = open(os.path.join(config.dataset.root_dir, "calibration.json"), "r")
        self.calib = json.load(f)
        self.center, self.scale = self.get_camera_center_scale()
        self.filenames = [file for file in os.listdir(self.data_dir) if file.endswith('.jpg')]
        self.width = config.dataset.img_width // config.dataset.downscale_factor
        self.height = config.dataset.img_height // config.dataset.downscale_factor

    def get_camera_center_scale(self):
        cameras = [elem for elem in self.calib['cameras'] if elem['type']=='hd']
        Ts = []
        for cam in cameras:
            Rc = torch.tensor(cam['R'])
            C = torch.squeeze(torch.tensor(cam['t']))
            R, t = local_to_world(Rc, C)
            Ts.append(t)
        Ts = torch.stack(Ts, dim=0)
        return get_center_scale(Ts)

    def __getitem__(self, index) -> any:
        filename = self.filenames[index]
        silhouette_image = cv2.imread(os.path.join(self.data_dir, filename), cv2.IMREAD_GRAYSCALE)
        #silhouette_image = cv2.resize(silhouette_image, dsize=(128,128))
        silhouette_image = cv2.resize(silhouette_image, dsize=(self.width, self.height))
        silhouette_tensor = torch.tensor(silhouette_image, dtype=torch.float).unsqueeze(0)
        silhouette_tensor = silhouette_tensor/255. # normalize to range [0, 1]
        # set all values above 0.5 to 1, all below 0.5 to 0
        silhouette_tensor = torch.where(silhouette_tensor > 0.5, torch.tensor(1.0), torch.tensor(0.0)) 
        
        camera_name = '_'.join(filename.split('_')[1:3])
        camera = [elem for elem in self.calib['cameras'] if elem['type']=='hd' and elem['name']==camera_name][0]
        
        # c2w, center & scale, w2c
        Rc = torch.tensor(camera['R'])
        C = torch.squeeze(torch.tensor(camera['t']))
        R, t = local_to_world(Rc, C)
        t -= self.center
        t *= self.scale
        Rc, C = world_to_local(R, t)

        item = {
            "silhouette": silhouette_tensor,
            "R": Rc.transpose(0,1),
            "t": C
        }

        if 'K' in camera:
            K = torch.eye(4)
            K[0:3,0:3] = torch.tensor(camera['K'])
            item["K"] = K
        return item
    
    def __len__(self):
        return len(self.filenames)
