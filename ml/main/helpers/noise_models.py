import numpy as np
import cv2

def gaussian_noise(img):
    """
    Add Gaussian noise to an image. The standard deviation is sampled
    randomly from [0, 50] as per the Noise2Noise paper.

    Args:
        img (numpy.ndarray): Input image with pixel values in the range [0, 255].

    Returns:
        numpy.ndarray: Noisy image with pixel values in the range [0, 255].
    """
    MIN_STDDEV = 0
    MAX_STDDEV = 50
    stddev = np.random.uniform(MIN_STDDEV, MAX_STDDEV)

    noisy_img = img.astype(np.float64)
    noise = np.random.normal(loc=0.0, scale=stddev, size=img.shape)
    noisy_img += noise

    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img
    
def main():
    noise_model = gaussian_noise

    while True:
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