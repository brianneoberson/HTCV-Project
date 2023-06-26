from omegaconf import OmegaConf
import argparse
from utils.generate_cow_renders import generate_cow_renders
from utils.read_camera_parameters import read_camera_parameters
from utils.create_target_images import create_target_images
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
)
import pytorch_lightning as pl
from dataloader import NerfDataset
from torch.utils.data import DataLoader
from models.nerf_light import Nerf

# -------------------------------------------------------------------------
#
# Arguments
#
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Config file containing all hyperparameters.')
args = parser.parse_args()

config = OmegaConf.load(args.config)

# -------------------------------------------------------------------------
#
# Data 
#
# dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
# mnist_train, mnist_val = random_split(dataset, [55000, 5000])
# train_loader = DataLoader(mnist_train, batch_size=32)
# val_loader = DataLoader(mnist_val, batch_size=32)

dataset = NerfDataset(config)
dataloader = DataLoader(dataset, batch_size=6)
model = Nerf(config)

# training
trainer = pl.Trainer(accelerator="gpu", precision=16, devices=1)
trainer.fit(model, dataloader)
