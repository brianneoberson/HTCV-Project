import os
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from models.nesc import NeSC
import torch.utils.data as data
from silhouette_dataset import SilhouetteDataset
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

# -------------------------------------------------------------------------

# output folder for logs, ckpts, .ply 
output_dir = os.path.join(config.output_dir)
if not os.path.exists(output_dir): 
    os.makedirs(output_dir)
tb_logger = pl_loggers.TensorBoardLogger(name=config.experiment_name, save_dir=output_dir)


# initialize checkpoint callback 
checkpoint_callback = ModelCheckpoint(
    every_n_epochs=config.checkpoint.save_every_n_epochs
    )

# training
pl.seed_everything(config.seed, workers=True)

# set model
match config.model.name:
    case "nesc":
        model = NeSC(config)
    case "nerf":
        model = NeRF(config)
    case _:
        raise ValueError("Not supported model name: {}".format(config.model.name))

# save config
if not os.path.exists(tb_logger.log_dir): 
    os.makedirs(tb_logger.log_dir)
with open(os.path.join(tb_logger.log_dir, "config.yaml"), "w") as f:
    OmegaConf.save(config, f)


dataset = SilhouetteDataset(config)
train_set_size = int(len(dataset) * config.dataset.training_split)
valid_set_size = len(dataset) - train_set_size

print(f'Training test size: {train_set_size}')
print(f'Validation test size: {valid_set_size}')

generator = torch.Generator().manual_seed(config.seed)
train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=generator)
train_loader = data.DataLoader(train_set, batch_size=config.trainer.batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_set, batch_size=config.trainer.batch_size, shuffle=True)

trainer = pl.Trainer(
    logger=tb_logger,
    max_epochs=config.trainer.max_epochs,
    accelerator=config.trainer.device, 
    devices=config.trainer.num_devices,
    callbacks=checkpoint_callback,
    log_every_n_steps=config.trainer.log_every_n_steps,
    check_val_every_n_epoch=config.trainer.check_val_every_n_epoch
    )

trainer.fit(model, train_loader, valid_loader)