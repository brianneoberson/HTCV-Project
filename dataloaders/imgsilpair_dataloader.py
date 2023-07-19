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

class ImgSilPairDataset(Dataset):
    def __init__(self, config) -> None:
        super().__init__()
        self.sil_dir = os.path.join(config.dataset.root_dir, "silhouettes")
        self.img_dir = os.path.join(config.dataset.root_dir, "extracted_frames")  
        f = open(os.path.join(config.dataset.root_dir, "calibration.json"), "r")
        self.cameras = json.load(f)
        self.sil_filenames = [file for file in os.listdir(self.sil_dir) if file.endswith('.jpg')]
        self.img_filenames = [file for file in os.listdir(self.img_dir) if file.endswith('.jpg')]

    def __getitem__(self, index) -> any:
        
        sil_filename = self.sil_filenames[index]
        img_filename = self.img_filenames[index]
        
        image = Image.open(os.path.join(self.img_dir, img_filename))
        image = image.resize((128,128))
        image_tensor = torch.tensor(np.array(image), dtype=torch.float).unsqueeze(0)
        
        sil = Image.open(os.path.join(self.sil_dir, sil_filename))
        sil = ImageOps.grayscale(sil)
        sil = sil.resize((128,128))
        silhouette_tensor = torch.tensor(np.array(sil), dtype=torch.float).unsqueeze(0)
        
        camera_name = '_'.join(sil_filename.split('_')[1:3]) ## can use either sil or img filepaths
        camera = [elem for elem in self.cameras['cameras'] if elem['type']=='hd' and elem['name']==camera_name][0]
        K = torch.eye(4)
        K[0:3,0:3] = torch.tensor(camera['K'])
        R = torch.tensor(camera['R'])
        t = torch.squeeze(torch.tensor(camera['t']))
        return image_tensor, silhouette_tensor, K, R, t
    
    def __len__(self):
        return len(self.sil_filenames)


class NerfDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.trainer.batch_size        

    def setup(self, stage: str):
        self.data = ImgSilPairDataset(self.config)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True)

    # not using yet:
    def val_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size)