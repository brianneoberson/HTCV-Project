import os.path

import numpy as np
import requests
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

from segmentation import generate_boxes


# url = os.path.join("extracted_frames", "talking_dog_frames", "00000100.jpg")
def BGRemove(url):
    image = Image.open(url)
    im, score, box, mask = generate_boxes(url=url)
    normImage = np.array(image) / 255

    selectedPixelsBg = normImage[mask == 0]
    selectedPixelsFg = normImage[mask == 1]

    fgGmm = GaussianMixture(n_components=50, verbose=2)
    fgGmm.fit(selectedPixelsFg)

    bgGmm = GaussianMixture(n_components=50, verbose=2)
    bgGmm.fit(selectedPixelsBg)
    fgLikelihood = fgGmm.score_samples(normImage.reshape(-1, 3))
    bgLikelihood = bgGmm.score_samples(normImage.reshape(-1, 3))

    segmentation = (fgLikelihood < bgLikelihood).reshape(normImage.shape[0:2])
    heatMap = (fgLikelihood - bgLikelihood).reshape(normImage.shape[0:2])
    plt.imshow(heatMap, cmap='hot', interpolation='nearest')
    plt.show()
    segmentation_expanded = np.repeat(segmentation[:, :, np.newaxis], 3, axis=2)

    backgroundMasked = np.where(segmentation_expanded, 0, normImage)
    foregroundMasked = np.where(segmentation_expanded, normImage, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[1].imshow(backgroundMasked)
    axes[1].set_title('Background Masked')
    axes[2].imshow(foregroundMasked)
    axes[2].set_title('Foreground Masked')
    plt.show()
#BGRemove(os.path.join("extracted_frames","hd_00_00_frames", "00000000.jpg"))
import cv2
#image, score, box, mask = generate_boxes(url=os.path.join("extracted_frames","hd_00_00_frames", "00000000.jpg"))
# Carica l'immagine
image = cv2.imread(os.path.join("extracted_frames","hd_00_00_frames", "00000000.jpg"))
# Converti l'immagine in scala di grigi
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applica il filtro di Canny per rilevare i bordi
edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Trova i contorni nell'immagine
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Disegna i contorni sull'immagine originale
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)

# Mostra l'immagine risultante
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()