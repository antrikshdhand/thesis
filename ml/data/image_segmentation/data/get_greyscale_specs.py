import numpy as np
import cv2
from pathlib import Path

image_dir = "spectrograms/"
all_image_paths = [p for p in Path(image_dir).glob("**/*")]
num_images = len(all_image_paths)

save_path = "greyscale_train/"
Path.mkdir(Path(save_path), exist_ok=True)

EDGE_SIZE = 180
processed_images = np.zeros(shape=(num_images, EDGE_SIZE, EDGE_SIZE))
image_counter = 0 
for img in all_image_paths:
    img_grey = cv2.imread(img, flags=cv2.IMREAD_GRAYSCALE) 

    if img_grey is None:
        continue
    
    h, w = img_grey.shape
    if h < EDGE_SIZE or w < EDGE_SIZE:
        continue

    # Randomly choose a patch
    i = np.random.randint(0, h - EDGE_SIZE + 1)
    j = np.random.randint(0, w - EDGE_SIZE + 1)
    patch = img_grey[i:i + EDGE_SIZE, j:j + EDGE_SIZE]

    processed_images[image_counter] = patch

    image_counter += 1

print(processed_images.shape)
processed_images = np.expand_dims(processed_images, axis=-1)

np.savez_compressed(f"{save_path}greyscale.npz", np_data=processed_images) 

