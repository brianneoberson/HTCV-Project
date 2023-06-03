import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import sys
import os

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def blackWitheSegmentation(mask_generator, mask_generator_2, image):
    # Get the first mask
    masks = mask_generator.generate(image)
    first_mask = masks[0]['segmentation']

    # Convert the mask to a binary mask where the object is white and the rest is black
    binary_mask = np.where(first_mask > 0, 255, 0).astype('uint8')

    # Create a blank 3-channel image with the same dimensions as the original image
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    # Assign the binary mask to each channel of the blank image
    blank_image[:, :, 0] = binary_mask
    blank_image[:, :, 1] = binary_mask
    blank_image[:, :, 2] = binary_mask

    # Display the binary mask image
    cv2.imshow('Binary Mask', blank_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    masks2 = mask_generator_2.generate(image)

    return blank_image


def mkDir(mask_generator, mask_generator_2):
    # Path to the directory containing the images
    input_dir = r"input"
    # Path to the directory where the segmented images will be saved
    output_dir = r'opt'
    # Path to the model checkpoint
    checkpoint_path = r'segment_anything\sam_vit_h_4b8939.pth'
    # Model type
    # model_type = 'vit_h'  # Choose from 'default', 'vit_h', 'vit_l', or 'vit_b'

    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model
    # sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

    # Iterate over all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # Add or modify the file extensions as needed
            # Load the image
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)  # Replace with the appropriate image loading function
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Generate the mask
            mask = blackWitheSegmentation(mask_generator, mask_generator_2, image)

            # Save the segmented image
            mask_path = os.path.join(output_dir, filename)
            cv2.imwrite(mask_path, mask)  # Replace with the appropriate mask saving function


if __name__ == "__main__":
    """
    image = cv2.imread('input/00000000.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    """
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    mkDir(mask_generator, mask_generator_2)

"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


if __name__ == "__main__":
    image = cv2.imread('00000000.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )


    # Get the first mask
    masks = mask_generator.generate(image)
    first_mask = masks[0]['segmentation']

    # Convert the mask to a binary mask where the object is white and the rest is black
    binary_mask = np.where(first_mask > 0, 255, 0).astype('uint8')

    # Create a blank 3-channel image with the same dimensions as the original image
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    # Assign the binary mask to each channel of the blank image
    blank_image[:, :, 0] = binary_mask
    blank_image[:, :, 1] = binary_mask
    blank_image[:, :, 2] = binary_mask

    # Display the binary mask image
    cv2.imshow('Binary Mask', blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    masks2 = mask_generator_2.generate(image)
    print(len(masks))
    print(masks[0].keys())
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks2)
    plt.axis('off')
    plt.show()"""

# %%
