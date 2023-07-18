import pytorch_lightning as pl
import os
from utils.read_camera_parameters import read_camera_parameters
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

class SilhouetteDataset(Dataset):
    def __init__(self, config) -> None:
        super().__init__()
        self.data_dir = config.dataset.root_dir
        f = open(os.path.join(self.data_dir, "calibration.json"), "r")
        self.cameras = json.load(f)
        self.filenames = [file for file in os.listdir(self.data_dir) if file.endswith('.jpg')]

    def __getitem__(self, index) -> any:
        filename = self.filenames[index]
        image = Image.open(os.path.join(self.data_dir, filename))
        image = ImageOps.grayscale(image)
        image = image.resize((128,128))
        silhouette_tensor = torch.tensor(np.array(image), dtype=torch.float).unsqueeze(0)
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