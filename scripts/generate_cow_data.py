from utils.generate_cow_renders import generate_cow_renders
import numpy as np
import os
import cv2
from tqdm import tqdm
import json

print("Loading data...")
target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=40, azimuth_range=180, data_dir="data/cow_mesh")

print("Creating folders...")
output_dir = "data/cow_data"
image_dir = os.path.join(output_dir, "images")
sil_dir = os.path.join(output_dir, "silhouettes")

if not os.path.exists(output_dir): 
    os.mkdir(output_dir)
if not os.path.exists(image_dir): 
    os.mkdir(image_dir)
if not os.path.exists(sil_dir): 
    os.mkdir(sil_dir)

calibration = {
    "cameras": []
}
for i in tqdm(range(target_images.shape[0])):
    filename = f"hd_00_{i:02}_cow.jpg"

    image = target_images[i].numpy()
    image = (image * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(image_dir, filename), image)

    silhouette = target_silhouettes[i].unsqueeze(-1).numpy()
    silhouette = (silhouette * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(sil_dir, filename), silhouette)

    camera = target_cameras[i]
    R = camera.R.squeeze()
    t = camera.T.squeeze()

    camera_dict = {
        "type": "hd",
        "name": f"00_{i:02}",
        "R": R.numpy().tolist(),
        "t": t.numpy().tolist()
    }
    calibration["cameras"].append(camera_dict)

print("Saving calibration file...")
calib_file = os.path.join(output_dir, "calibration.json")
with open(calib_file, 'w') as file:
    json.dump(calibration, file, indent=4)

print("Done.")
