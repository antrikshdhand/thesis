import pandas as pd
import numpy as np
import keras
import os
from typing import Literal, Optional, List, Callable
from pathlib import Path
import random
import cv2

from helpers.data import import_spectrogram
from helpers.synthetic_spectrograms import *


class DeepShipGenerator(keras.utils.Sequence):
    """
    Generator for DeepShip dataset, used for either classification or 
    autoencoder-based tasks.
    
    - For classification tasks, it provides (X, y) pairs where X is a spectrogram 
        and y is a class label.
    - For autoencoder denoising tasks, it provides (X, X) pairs, treating the 
        same spectrogram X as both input and output.
    """

    def __init__(self, 
                 df: pd.DataFrame, 
                 ext: Literal['csv', 'npz', 'mat'], 
                 mat_var_name: Optional[str] = None, 
                 classes: List[str] = ['Cargo', 'Passengership', 'Tanker', 'Tug'],
                 batch_size: int = 32, 
                 shuffle: bool = True, 
                 conv_channel: bool = True, 
                 zero_one_normalised: bool = False,
                 X_only: bool = False):
        """
        Initialise the generator.

        :param df: DataFrame containing file paths and labels.
        :param ext: File extension for spectrogram files ('mat', 'csv', or 'npz').
        :param mat_var_name: Variable name in .mat files if applicable.
        :param classes: List of class labels for one-hot encoding (for classification tasks).
        :param batch_size: Size of each batch.
        :param shuffle: Whether to shuffle data at the end of each epoch.
        :param conv_channel: Whether to add a channel dimension for CNN input.
        :param zero_one_normalised: Whether to normalise spectrograms between [0, 1].
        :param X_only: If True, outputs (X, X) for autoencoder tasks; otherwise, (X, y).
        """

        self.df = df
        self.ext = ext
        self.mat_var_name = mat_var_name
        self.classes = classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.conv_channel = conv_channel
        self.zero_one_normalised = zero_one_normalised
        self.X_only = X_only

        self.n = len(self.df)
        
        # Shuffle the data initially
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return self.n // self.batch_size

    def __get_x(self, batch_df):
        """
        Load and preprocess the spectrograms for the batch.
        """
        # Import spectrograms for each file in the batch
        batch_df = import_spectrogram(batch_df, self.ext, self.mat_var_name,
                                      source_col='files', dest_col='spectrogram')

        X = np.stack(batch_df['spectrogram'].to_numpy(copy=True))

        X_min = X.min(axis=(1, 2), keepdims=True)
        X_max = X.max(axis=(1, 2), keepdims=True)
        X = (X - X_min) / (X_max - X_min) # normalised

        # Add channel dimension if using convolutional layers
        if self.conv_channel:
            X = np.expand_dims(X, axis=-1)

        return X

    def __get_y(self, batch_df):
        """
        Get one-hot encoded labels for the batch (for classification tasks).
        """
        return batch_df[self.classes].to_numpy(copy=True)

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        # Select the batch data
        batch_df = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.__get_x(batch_df)
        y = self.__get_y(batch_df)

        if self.X_only:
            return X, X
        else:
            return X, y

    def on_epoch_end(self):
        """
        Shuffle the data at the end of each epoch if specified.
        """
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)


class N2NGenerator(keras.utils.Sequence):
    """
    Noise2Noise generator for denoising tasks where input (X) and output (y) spectrograms 
    are different but come from similar recordings. Suitable for tasks where we want the model 
    to learn to denoise or uncover important features by observing slightly varied spectrograms.
    """

    def __init__(self, 
                 df: pd.DataFrame, 
                 ext: Literal['csv', 'npz', 'mat'], 
                 mat_var_name: Optional[str] = None, 
                 batch_size: int = 32, 
                 shuffle: bool = True, 
                 conv_channel: bool = True,
                 **kwargs):    
        """
        Initialise the generator.

        :param df: DataFrame containing file paths for pairs of noisy spectrograms.
        :param ext: File extension for spectrogram files ('mat', 'csv', or 'npz').
        :param mat_var_name: Variable name in .mat files if applicable.
        :param batch_size: Size of each batch.
        :param shuffle: Whether to shuffle data at the end of each epoch.
        :param conv_channel: Whether to add a channel dimension for CNN input.
        """
        self.df = df 
        self.ext = ext
        self.mat_var_name = mat_var_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.conv_channel = conv_channel

        self.n = len(df)

        if self.shuffle:
            self.on_epoch_end()
        
        super().__init__(**kwargs)

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return self.n // self.batch_size

    def __get_data(self, batch_df):
        """
        Load and preprocess both X and y spectrograms for the batch. 
        X and y are different spectrograms from the same vessel, used for denoising.
        """

        # First convert X spectrogram
        batch_df = import_spectrogram(batch_df, ext=self.ext, mat_var_name=self.mat_var_name,
                                      source_col='file_path_1', dest_col='spec_1')
        # Then convert y spectrogram
        batch_df = import_spectrogram(batch_df, ext=self.ext, mat_var_name=self.mat_var_name,
                                      source_col='file_path_2', dest_col='spec_2')
        
        # Resize to fit U-Net
        batch_df['spec_1'] = batch_df['spec_1'].apply(lambda x: cv2.resize(x, (256, 256)))
        batch_df['spec_2'] = batch_df['spec_2'].apply(lambda x: cv2.resize(x, (256, 256)))

        X = np.stack(batch_df['spec_1'].to_numpy(copy=True))
        y = np.stack(batch_df['spec_2'].to_numpy(copy=True))

        X = np.repeat(np.expand_dims(X, axis=-1), 3, axis=-1)
        y = np.repeat(np.expand_dims(y, axis=-1), 3, axis=-1)
        
        return X, y

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """

        batch_df = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]

        return self.__get_data(batch_df)

    def on_epoch_end(self):
        """
        Shuffle the data at the end of each epoch.
        """
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)


class ImageSegmentationGenerator(keras.utils.Sequence):
    """
    Loads batches of spectrograms and masks for image segmentation training.
    """

    def __init__(self, specs_dir: str, masks_dir: str, spec_files: List[str],
                 mask_files: List[str], batch_size: int,
                 image_dims: tuple = (256, 256), shuffle: bool = True):
        """
        Initialise generator.

        :param str specs_dir: Directory of spectrogram images.
        :param str masks_dir: Directory of mask images.
        :param list spec_files: List of spectrogram filenames.
        :param list mask_files: List of mask filenames.
        :param int batch_size: Number of samples per batch.
        :param tuple image_dims: Target dimensions for resizing.
        :param bool shuffle: Whether to shuffle data at the end of each epoch.
        """
        self.specs_dir = specs_dir
        self.masks_dir = masks_dir
        self.spec_files = spec_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.image_dims = image_dims
        self.shuffle = shuffle
        self.n = len(self.spec_files)

        self.on_epoch_end()

    def __len__(self):
        return self.n // self.batch_size

    def __get_data(self, batch_spec_files, batch_mask_files):
        """
        Loads and preprocesses a batch of spectrograms and masks.
        :param batch_spec_files: List of spectrogram file paths for the batch.
        :param batch_mask_files: List of mask file paths for the batch.
        :return: Tuple of (specs, masks) as numpy arrays.
        """
        specs = []
        masks = []

        for spec_file, mask_file in zip(batch_spec_files, batch_mask_files):
            spec = keras.utils.load_img(
                os.path.join(self.specs_dir, spec_file), 
                color_mode="grayscale", 
                target_size=self.image_dims
            )
            mask = keras.utils.load_img(
                os.path.join(self.masks_dir, mask_file), 
                color_mode="grayscale", 
                target_size=self.image_dims
            )

            # Convert to numpy arrays [0, 1] where 1 is label
            spec = keras.utils.img_to_array(spec) / 255.0
            mask = keras.utils.img_to_array(mask) / 255.0

            # Append to lists
            specs.append(spec)
            masks.append(np.round(mask)) # Ensure binary masks
        
        return np.array(specs, dtype=np.float32), np.array(masks, dtype=np.float32)

    def __getitem__(self, index):
        """
        Generates one batch of data.
        :param index: Index of the batch.
        :return: Tuple of (specs, masks) for the batch.
        """

        batch_spec_files = self.spec_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_mask_files = self.mask_files[index * self.batch_size:(index + 1) * self.batch_size]

        return self.__get_data(batch_spec_files, batch_mask_files)
    
    def on_epoch_end(self):
        """
        Shuffle data at the end of each epoch if `shuffle` is set to True.
        """
        if self.shuffle:
            temp = list(zip(self.spec_files, self.mask_files))
            np.random.shuffle(temp)
            self.spec_files, self.mask_files = zip(*temp)
            self.spec_files = list(self.spec_files)
            self.mask_files = list(self.mask_files)


class NoisyImageTrainGenerator(keras.utils.Sequence):
    """
    Given a directory of images, this generator outputs X, y arrays of size
    `batch_size` both containing the same images under two different noise
    models `input_noise_model` and `target_noise_model`.

    i.e. Outputs (noisy1, noisy2) pairs.
    """

    def __init__(self, image_dir: str, input_noise_model: Callable, 
                target_noise_model: Callable, batch_size: int = 16, 
                patch_edge_size: int = 256, **kwargs):

        image_suffixes = ['.jpeg', '.png', '.jpg']
        self.all_image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() 
                                in image_suffixes]
        self.num_images = len(self.all_image_paths)
        if self.num_images == 0:
            raise ValueError(f"The given directory {image_dir} does not contain any images.")

        self.input_noise_model = input_noise_model
        self.target_noise_model = target_noise_model
        self.batch_size = batch_size
        self.patch_edge_size = patch_edge_size 

        super().__init__(**kwargs)

    def __len__(self):
        return self.num_images // self.batch_size 

    def __getitem__(self, idx):
        batch_size = self.batch_size
        edge_size = self.patch_edge_size
        X = np.zeros((batch_size, edge_size, edge_size, 3), dtype=np.float32)
        y = np.zeros((batch_size, edge_size, edge_size, 3), dtype=np.float32)

        image_counter = 0
        while image_counter < batch_size:
            image_path = random.choice(self.all_image_paths)
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            h, w, _ = image.shape
            if h < edge_size or w < edge_size:
                continue

            # Randomly choose a patch
            i = np.random.randint(0, h - edge_size + 1)
            j = np.random.randint(0, w - edge_size + 1)
            patch = image[i:i + edge_size, j:j + edge_size]

            # Add noise and normalize
            input_patch = self.input_noise_model(patch.astype(np.float32)) / 255.0
            target_patch = self.target_noise_model(patch.astype(np.float32)) / 255.0

            # Store patches
            X[image_counter] = input_patch
            y[image_counter] = target_patch

            image_counter += 1

        return X, y
    

class NoisyImageValGenerator(keras.utils.Sequence):
    """
    Outputs (noisy, clean) pairs for validation.
    """

    def __init__(self, image_dir: str, val_noise_model: Callable, batch_size: int = 16, 
                 patch_edge_size: int = 256, **kwargs):

        image_suffixes = ['.jpeg', '.png', '.jpg']
        self.all_image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() 
                                in image_suffixes]
        self.num_images = len(self.all_image_paths)
        if self.num_images == 0:
            raise ValueError(f"The given directory {image_dir} does not contain any images.")
        
        self.val_noise_model = val_noise_model
        self.batch_size = batch_size
        self.patch_edge_size = patch_edge_size 

        super().__init__(**kwargs)

    def __len__(self):
        return self.num_images // self.batch_size 

    def __getitem__(self, idx):
        batch_size = self.batch_size
        edge_size = self.patch_edge_size
        X = np.zeros((batch_size, edge_size, edge_size, 3), dtype=np.float32)
        y = np.zeros((batch_size, edge_size, edge_size, 3), dtype=np.float32)

        image_counter = 0
        while image_counter < batch_size:
            image_path = random.choice(self.all_image_paths)
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            h, w, _ = image.shape
            if h < edge_size or w < edge_size:
                continue

            # Randomly choose a patch
            i = np.random.randint(0, h - edge_size + 1)
            j = np.random.randint(0, w - edge_size + 1)
            clean_patch = image[i:i + edge_size, j:j + edge_size]

            # Add noise and normalize
            noisy_patch = self.val_noise_model(clean_patch.astype(np.float32)) / 255.0
            clean_patch = clean_patch.astype(np.float32) / 255.0

            X[image_counter] = noisy_patch
            y[image_counter] = clean_patch

            image_counter += 1

        return X, y


class SyntheticSpectrogramGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, num_batches, fs, duration, window, noverlap, nfft, **kwargs):
        self.batch_size = batch_size
        self._num_batches = num_batches 
        self.fs = fs
        self.duration = duration
        self.window = window
        self.noverlap = noverlap
        self.nfft = nfft

        super().__init__(**kwargs)

    def __len__(self):
        return self._num_batches

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, 256, 256, 3))  # Updated to 3 channels
        y = np.zeros((self.batch_size, 256, 256, 3))  # Updated to 3 channels

        image_counter = 0
        for _ in range(self.batch_size):
            _, clean_signal, _ = generate_sine_wave(self.duration, self.fs)

            noisy_signal_1, _ = add_gaussian_noise(clean_signal)
            noisy_signal_2, _ = add_gaussian_noise(clean_signal)

            # Generate spectrograms
            _, _, noisy_spectrogram_1 = create_spectrogram(noisy_signal_1, self.fs, self.window, self.noverlap, self.nfft)
            _, _, noisy_spectrogram_2 = create_spectrogram(noisy_signal_2, self.fs, self.window, self.noverlap, self.nfft)

            # Normalize spectrograms for neural network input
            noisy_spectrogram_1 = np.log1p(noisy_spectrogram_1)
            noisy_spectrogram_2 = np.log1p(noisy_spectrogram_2)

            # Resize for U-Net dimensions
            noisy_spectrogram_1 = cv2.resize(noisy_spectrogram_1, (256, 256))
            noisy_spectrogram_2 = cv2.resize(noisy_spectrogram_2, (256, 256))

            # Expand to 3 channels
            noisy_spectrogram_1 = np.repeat(np.expand_dims(noisy_spectrogram_1, axis=-1), 3, axis=-1)
            noisy_spectrogram_2 = np.repeat(np.expand_dims(noisy_spectrogram_2, axis=-1), 3, axis=-1)

            X[image_counter] = noisy_spectrogram_1
            y[image_counter] = noisy_spectrogram_2

            image_counter += 1

        return X, y