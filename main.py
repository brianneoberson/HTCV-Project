import os
import argparse
import yaml
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from models.nerf_light import Nerf
from models.nerf_color import NerfColor
import torch
torch.set_float32_matmul_precision('high')

# -------------------------------------------------------------------------
#
# Arguments
#
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file containing all hyperparameters.')
args = parser.parse_args()

config = OmegaConf.load(args.config)

print()
print(config)
# -------------------------------------------------------------------------

# output folder for logs, ckpts, .ply 
output_dir = os.path.join(config.output_dir, config.experiment_name)
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)
tb_logger = pl_loggers.TensorBoardLogger(save_dir=output_dir)

## debug image logging ##
from torch.utils.data import DataLoader
from dataloaders.silhouette_dataloader import SilhouetteDataset

if not config["dataset"]["cow"]:
    dataset = SilhouetteDataset(config)
    test_image, _ , _ , _ = dataset.__getitem__(3)
    tb_logger.experiment.add_image("test image", test_image)


# initialize checkpoint callback 
checkpoint_callback = ModelCheckpoint(
    every_n_epochs=config.checkpoint.save_every_n_epochs
    )

# training
pl.seed_everything(config.seed, workers=True)

# set model
match config.model.name:
    case "nerf":
        model = Nerf(config)
    case "nerf_color":
        model = NerfColor(config)
    case _:
        raise ValueError("Not supported model name: {}".format(config.model.name))
        
trainer = pl.Trainer(
    logger=tb_logger,
    max_epochs=config.trainer.max_epochs,
    accelerator=config.trainer.device, 
    devices=config.trainer.num_devices,
    check_val_every_n_epoch=1,
    callbacks=checkpoint_callback,
    log_every_n_steps=config.trainer.log_every_n_steps,
    )

# save config
with open(os.path.join(tb_logger.log_dir, "config.yaml"), "w") as f:
    OmegaConf.save(config, f)

trainer.fit(model)
