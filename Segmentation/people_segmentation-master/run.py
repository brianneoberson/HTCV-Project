import os

import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model


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
    model = create_model("Unet_2020-07-20")
    model.eval()
    input_dir = r"input\extracted_frames_todler"
    output_dir = r'opt'
    for filename in os.listdir(input_dir):
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
