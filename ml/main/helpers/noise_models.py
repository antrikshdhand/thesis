import numpy as np
import cv2

def gaussian_noise(img, stddev=None):
    """
    Add Gaussian noise to an image. The standard deviation is sampled
    randomly from [0, 50] as per the Noise2Noise paper. Automatically handles
    whether the image is normalised or not based on its range.
    """
    MIN_STDDEV = 0
    MAX_STDDEV = 50
    if stddev is None:
        stddev = np.random.uniform(MIN_STDDEV, MAX_STDDEV)

    noisy_img = img.astype(np.float64)
    noise = np.random.normal(loc=0.0, scale=stddev, size=img.shape)

    if noisy_img.max() <= 1.0:
        # Image is normalised
        noisy_img += noise / 255.0
        noisy_img = np.clip(noisy_img, 0, 1)
    else:
        # Image is in [0, 255] range
        noisy_img += noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img
    
def main():
    noise_model = gaussian_noise

    image = np.ones((256, 256, 3), dtype=np.uint8)
    noisy_image = noise_model(image)
    cv2.imshow('Clean image', image)
    cv2.imshow('Noisy image', noisy_image)
    key = cv2.waitKey(-1)

    # "q": quit
    if key == 113:
        return 0


if __name__ == '__main__':
    main()