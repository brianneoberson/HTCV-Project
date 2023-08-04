import pytorch_lightning as pl
import os
from utils.camera_utils import read_camera_parameters
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
        self.cameras = json.load(f)
        self.filenames = [file for file in os.listdir(self.data_dir) if file.endswith('.jpg')]

    def __getitem__(self, index) -> any:
        filename = self.filenames[index]
        silhouette_image = cv2.imread(os.path.join(self.data_dir, filename), cv2.IMREAD_GRAYSCALE)
        silhouette_image = cv2.resize(silhouette_image, dsize=(128,128))
        silhouette_tensor = torch.tensor(silhouette_image, dtype=torch.float).unsqueeze(0)
        silhouette_tensor = silhouette_tensor/255. # normalize to range [0, 1]
        # set all values above 0.5 to 1, all below 0.5 to 0
        silhouette_tensor = torch.where(silhouette_tensor > 0.5, torch.tensor(1.0), torch.tensor(0.0)) 
        
        camera_name = '_'.join(filename.split('_')[1:3])
        camera = [elem for elem in self.cameras['cameras'] if elem['type']=='hd' and elem['name']==camera_name][0]
        K = torch.eye(4)
        K[0:3,0:3] = torch.tensor(camera['K'])
        R = torch.tensor(camera['R'])
        t = torch.squeeze(torch.tensor(camera['t']))
        return silhouette_tensor, K, R, t
    
    def __len__(self):
        return len(self.filenames)


class NerfDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.trainer.batch_size        

    def setup(self, stage: str):
        self.data = SilhouetteDataset(self.config)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    # not using yet:
    def val_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)