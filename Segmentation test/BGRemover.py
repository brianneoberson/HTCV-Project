from __future__ import print_function

import argparse
import os

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn.functional import interpolate
from matplotlib import pyplot as plt
from PIL import Image
import cv2 as cv

from BackgroundMattingV2.model import MattingRefine

class BGRemover:
    def __init__(self):
        self.device = torch.device('cuda')
        self.precision = torch.float32
        self.model = MattingRefine(backbone='mobilenetv2',
                                   backbone_scale=0.25,
                                   refine_mode='sampling',
                                   refine_sample_pixels=80_000)
        self.model.load_state_dict(torch.load(os.path.join("BackgroundMattingV2", "PyTorch", "pytorch_mobilenetv2.pth")))
        self.model = self.model.eval().to(self.precision).to(self.device)

    def remove(self, src, bgr):
        src = src.to(self.device)
        bgr = bgr.to(self.device)
        bgr = interpolate(bgr, size=src.shape[-2:], mode='bilinear', align_corners=False)
        pha, fgr = self.model(src, bgr)[:2]
        return pha, fgr

def process_frame(frame, fgMask, bg_remover):
    # Convert the frame and fgMask to PIL Image
    frame_image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    fgMask_image = Image.fromarray(fgMask).convert("RGB")

    # Convert the images to tensors
    frame_tensor = to_tensor(frame_image).unsqueeze(0).float()
    fgMask_tensor = to_tensor(fgMask_image).unsqueeze(0).float()

    # Apply the background removal
    pha, fgr = bg_remover.remove(frame_tensor, fgMask_tensor)

    # Convert the tensors back to PIL Images for displaying
    frame_image = to_pil_image(frame_tensor.squeeze(0))
    fgMask_image = to_pil_image(fgMask_tensor.squeeze(0))
    pha_image = to_pil_image(pha.squeeze(0))
    fgr_image = to_pil_image(fgr.squeeze(0))
    return frame_image, fgMask_image, pha_image, fgr_image
    """
    # Display the images
    frame_image.show()
    fgMask_image.show()
    pha_image.show()
    fgr_image.show()"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                  OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args()

    backSub = cv.createBackgroundSubtractorKNN()

    capture = cv.VideoCapture(os.path.join("demo_videos", "talking_dog.mp4"))
    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)

    bg_remover = BGRemover()
    ret, first_frame = capture.read()
    if not ret:
        print('Unable to read the first frame.')
        exit(0)

    # Convert the first frame to PIL Image
    background_image = Image.fromarray(cv.cvtColor(first_frame, cv.COLOR_BGR2RGB))

    # ... (resto del codice) ...

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        # Convert the frame and masks to PIL Images
        frame_image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        fgMask_image = Image.fromarray(fgMask).convert("RGB")

        # Convert the images to tensors
        frame_tensor = to_tensor(frame_image).unsqueeze(0).float()
        fgMask_tensor = to_tensor(fgMask_image).unsqueeze(0).float()
        bg_image_tensor = to_tensor(background_image).unsqueeze(0).float()

        pha, fgr = bg_remover.remove(frame_tensor, bg_image_tensor)

        # Convert the tensors back to PIL Images for displaying
        pha_image = to_pil_image(pha.squeeze(0))
        fgr_image = to_pil_image(fgr.squeeze(0))

        # Convert PIL Images to NumPy arrays
        pha_np = np.array(pha_image)
        fgr_np = np.array(fgr_image)

        # Normalize pixel values to [0, 1] range
        pha_np = pha_np / 255.0
        fgr_np = fgr_np / 255.0

        # Expand pha to 3 channels
        pha_np = np.dstack([pha_np, pha_np, pha_np])

        # Combine pha (foreground mask) and fgr (foreground image) to get the image without background
        no_bg_image_np = (pha_np * fgr_np * 255).astype(np.uint8)

        # Convert RGB to BGR
        no_bg_image_np = cv.cvtColor(no_bg_image_np, cv.COLOR_RGB2BGR)

        # Display the images
        cv.imshow('Frame', frame)
        cv.imshow('Foreground Mask', fgMask)
        #cv.imshow('Background Mask', background_image)
        cv.imshow('No Background Image', no_bg_image_np)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
