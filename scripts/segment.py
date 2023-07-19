import os
from collections import namedtuple
import numpy as np
import argparse
from tqdm import tqdm
import cv2
from os import walk
import re
import argparse
import json
import torch
from torch import nn
import albumentations as albu
from torch.utils import model_zoo
from iglovikov_helper_functions.dl.pytorch.utils import rename_layers
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from segmentation_models_pytorch import Unet

model = namedtuple("model", ["url", "model"])

models = {
    "Unet_2020-07-20": model(
        url="https://github.com/ternaus/people_segmentation/releases/download/0.0.1/2020-09-23a.zip",
        model=Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None),
    )
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)
    return model


def blackWitheSegmentation(mask, image):
    imageMasked = image
    for x in range(imageMasked.shape[0]):
        for y in range(imageMasked.shape[1]):
            if mask[x][y] == 0:
                imageMasked[x][y] = np.array([0, 0, 0])
            else:
                imageMasked[x][y] = np.array([255,255,255])
    return imageMasked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of input images to segment.')
    args = parser.parse_args()

    model = create_model("Unet_2020-07-20")
    model.eval()
    input_dir = args.input_dir
    output_dir = os.path.join(input_dir, "../silhouettes")
    if not os.path.exists(output_dir): 
        os.mkdir(output_dir)

    print("#. Segment images from dir : {}".format(input_dir))
    print("#. Store masks under dir   : {}".format(output_dir))
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            image = load_rgb(img_path)
            transform = albu.Compose([albu.Normalize(p=1)], p=1)
            padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
            x = transform(image=padded_image)["image"]
            x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
            with torch.no_grad():
                prediction = model(x)[0][0]
            mask = (prediction > 0).cpu().numpy().astype(np.uint8)
            mask = unpad(mask, pads)
            # imshow(mask)
            dst = cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)
            opt_path = os.path.join(output_dir, filename)
            cv2.imwrite(opt_path, blackWitheSegmentation(mask, dst))

    print("Done!")