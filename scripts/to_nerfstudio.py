import os
import json
import numpy as np
import cv2

# load calibration file
f = open("/home/brianne/HTCV-Project/data/150821_dance285/calibration.json", "r")
calibration_dict = json.load(f)
images = sorted(os.listdir('/home/brianne/HTCV-Project/data/150821_dance285/images'))
masks = sorted(os.listdir('/home/brianne/HTCV-Project/data/150821_dance285/silhouettes'))
transforms_dict = {
    "frames": []
}
hd_frames = [f for f in calibration_dict["cameras"] if f["type"] == "hd"]
for (frame, img, mask) in zip(hd_frames, images, masks):    
    resolution = frame["resolution"]
    R = np.array(frame['R'])
    t = np.array(frame['t'])
    k = np.array(frame['K'])
    fx = k[0][0]
    fy = k[1][1]
    cx = k[0][2]
    cy = k[1][2]
    transform = np.identity(4)
    transform[0:3,0:3] = R
    transform[0:3, 3] = t.squeeze()
    transform = np.linalg.inv(transform) 
    transform[0:3, 1:3] *= -1
    w = resolution[0]
    h = resolution[1]
    img_path = img
    mask_path = mask
    transforms_dict["frames"].append({
        "file_path": os.path.join("images", img),
        "transform_matrix": transform.tolist(),
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "mask_path": os.path.join("masks", mask)
    })

transforms = open("/home/brianne/HTCV-Project/data/150821_dance285/transforms.json", "w")
json.dump(transforms_dict, transforms, indent=4)

# save silhouettes as grayscale for nerfstudio
os.makedirs("/home/brianne/HTCV-Project/data/150821_dance285/masks", exist_ok=True)
masks_dir = "/home/brianne/HTCV-Project/data/150821_dance285/masks"
for mask in masks:
    mask_path = os.path.join("/home/brianne/HTCV-Project/data/150821_dance285/silhouettes", mask)
    sil = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(masks_dir, mask), sil)