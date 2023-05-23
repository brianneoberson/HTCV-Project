import os
import json
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from cameraParser import readCameraParameters

class SilhouetteDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split

        # TODO:
        self.target_cameras = readCameraParameters(self.config['camera_calibration_path'])
        self.target_silhouettes = 0
        

class SilhouetteDataset(Dataset, SilhouetteDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.target_silhouettes)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class SilhouetteIterableDataset(IterableDataset, SilhouetteDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}

class SilhouetteDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = SilhouetteIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = SilhouetteDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = SilhouetteDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = SilhouetteDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       
