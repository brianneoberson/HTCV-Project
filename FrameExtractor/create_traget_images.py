import os
from PIL import Image, ImageOps
import torch
import numpy as np


def create_target_images(dir_path):
    target_images=None
    # Iterate through the directory
    for filename in os.listdir(dir_path):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image using PIL
            image = Image.open(os.path.join(dir_path, filename))
            image = ImageOps.grayscale(image)
            image=image.resize((128,128))
            tensor = torch.tensor(np.array(image), dtype=torch.float).unsqueeze(0)

            if target_images==None:
                target_images=tensor
            else:
                target_images=torch.cat((target_images,tensor),dim=0)

    return target_images

def main(dir_path):
    tensors=create_target_images(dir_path=dir_path)
    print(tensors.size())

# Set the directory path
dir_path = "./Segmentation/segment-anything-main/opt"
main(dir_path)
            