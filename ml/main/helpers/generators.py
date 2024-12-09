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


'''
-- DEEPSHIP --
1. DeepShipGenerator - classification or autoencoder-based tasks on DeepShip spectrograms.
        Outputs (X, y) = (spectrogram, one-hot encoded label)
                (X, X) = (spectrogram, spectrogram)

-- UNSUPERVISED TECHNIQUES -- 
2. N2NDeepShipGenerator - outputs two spectrograms from the same vessel but different recordings.
        Outputs (X, X') = (seg from recording i, seg from recording j)
        UNDER DEVELOPMENT
3. N2NTrainGenerator - training generator for recreating Noise2Noise on natural images.
        Outputs (x, x') = (noisy1, noisy2) of the same image.

-- SUPERVISED TECHNIQUES -- 
4. SupervisedDenoisingGenerator - validation generator for recreating Noise2Noise on natural images.
        Outputs (x, y) = (noisy, clean)
5. SyntheticSpectrogramGenerator - creates a synthetic spectrogram and adds artificial noise on top.
        Outputs (x, y) = (noisy, clean)

-- NOT TESTED -- 
6. ImageSegmentationGenerator
'''

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
                 X_only: bool = False,
                 **kwargs):
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
        
        # Shuffle the data initially, if requested
        self.on_epoch_end()

        super().__init__(**kwargs)

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


class N2NDeepShipGenerator(keras.utils.Sequence):
    """
    Noise2Noise generator for denoising tasks where input (X) and output (y) spectrograms 
    are different but come from similar recordings. 
    """

    def __init__(self, 
                 multiple_recordings_df: pd.DataFrame,
                 ext: Literal['csv', 'npz', 'mat'], 
                 mat_var_name: Optional[str] = None, 
                 batch_size: int = 32, 
                 shuffle: bool = True, 
                 conv_channel: bool = True,
                 **kwargs):    
        """
        Initialise the generator.

        :param multiple_recordings_df: A dataframe containing all segments whose 
            ships have multiple recordings.
        :param ext: File extension for spectrogram files ('mat', 'csv', or 'npz').
        :param mat_var_name: Variable name in .mat files if applicable.
        :param batch_size: Size of each batch.
        :param shuffle: Whether to shuffle data at the end of each epoch.
        :param conv_channel: Whether to add a channel dimension for CNN input.
        """
        self.multiple_recordings_df = multiple_recordings_df
        self.ext = ext
        self.mat_var_name = mat_var_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.conv_channel = conv_channel
        
        super().__init__(**kwargs)

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return len(self.multiple_recordings_df) // self.batch_size

    def __get_data(self, batch_df):
        """
        Load and preprocess both X and y spectrograms for the batch. 
        X and y are different spectrograms from the same vessel, used for denoising.
        """

        # First convert X spectrogram
        batch_df = import_spectrogram(batch_df, ext=self.ext, 
                                      mat_var_name=self.mat_var_name,
                                      source_col='original_spectrogram', 
                                      dest_col='spec_1')
        # Then convert y spectrogram
        batch_df = import_spectrogram(batch_df, ext=self.ext, 
                                      mat_var_name=self.mat_var_name,
                                      source_col='paired_spectrogram', 
                                      dest_col='spec_2')

        X = np.stack(batch_df['spec_1'].to_numpy(copy=True))
        y = np.stack(batch_df['spec_2'].to_numpy(copy=True))

        X = np.expand_dims(X, axis=-1)
        y = np.expand_dims(y, axis=-1)

        X_min = X.min(axis=(1, 2), keepdims=True)
        X_max = X.max(axis=(1, 2), keepdims=True)
        X = (X - X_min) / (X_max - X_min) # normalised

        y_min = y.min(axis=(1, 2), keepdims=True)
        y_max = y.max(axis=(1, 2), keepdims=True)
        y = (y - y_min) / (y_max - y_min) # normalised

        X = X.astype(np.float32)  
        y = y.astype(np.float32)

        return X, y

    def _get_possible_pairings(self, ship_name, date):
        # Any segment which has the same ship name but NOT the same date is valid
        return self.multiple_recordings_df[
            (self.multiple_recordings_df["ship_name"] == ship_name) & 
            (self.multiple_recordings_df["date"] != date)
        ]

    def _assign_pairs(self, batch_segments):
        batch_specs = {
            "original_spectrogram": [],
            "paired_spectrogram": []
        }
        for _, row in batch_segments.iterrows():
            batch_specs["original_spectrogram"].append(row["file_path"])
            possible_pairings = self._get_possible_pairings(row["ship_name"], row["date"])
            if possible_pairings.empty:
                raise ValueError("NO PAIRINGS FOUND!")
            else:
                random_choice = possible_pairings.sample(1)

            batch_specs["paired_spectrogram"].append(random_choice["file_path"].iloc[0])
        
        return pd.DataFrame(batch_specs)

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """

        # 1. Get BATCH_SIZE rows from the multiple_recordings_df
        batch_segments = self.multiple_recordings_df.iloc[index * self.batch_size:(index + 1) * self.batch_size]

        # 2. For each segment (row) in the batch, we assign it a
        #       segment which is from a different recording of the same ship.
        batch_df = self._assign_pairs(batch_segments)

        # 3. Import both spectrograms
        return self.__get_data(batch_df)


class N2NTrainGenerator(keras.utils.Sequence):
    """
    Given a directory of images, this generator outputs X, y arrays of size
    `batch_size` both containing the same images under two different noise
    models `input_noise_model` and `target_noise_model`.

    i.e. Outputs (noisy1, noisy2) pairs.
    """

    def __init__(self, 
                 image_dir: str, 
                 input_noise_model: Callable, target_noise_model: Callable, 
                 batch_size: int = 16, 
                 patch_edge_size: int = 192,
                 zero_one_normalisation: bool = True,
                 greyscale: bool = True,
                 **kwargs):

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
        self.zero_one_normalisation = zero_one_normalisation
        self.greyscale = greyscale

        super().__init__(**kwargs)

    def __len__(self):
        return self.num_images // self.batch_size 

    def __getitem__(self, idx):
        batch_size = self.batch_size
        edge_size = self.patch_edge_size

        channels = 1 if self.greyscale else 3
        X = np.zeros((batch_size, edge_size, edge_size, channels), dtype=np.float32)
        y = np.zeros((batch_size, edge_size, edge_size, channels), dtype=np.float32)

        image_counter = 0
        while image_counter < batch_size:
            image_path = random.choice(self.all_image_paths)
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            if self.greyscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=-1)  # Add channel dimension

            # Normalise if requested
            if self.zero_one_normalisation:
                image = image.astype(np.float32) / 255.0

            h, w, _ = image.shape
            if h < edge_size or w < edge_size:
                continue

            # Randomly choose a patch
            i = np.random.randint(0, h - edge_size + 1)
            j = np.random.randint(0, w - edge_size + 1)
            patch = image[i:i + edge_size, j:j + edge_size]

            # Add noise 
            input_patch = self.input_noise_model(patch.astype(np.float32))
            target_patch = self.target_noise_model(patch.astype(np.float32))

            # Store patches
            X[image_counter] = input_patch
            y[image_counter] = target_patch

            image_counter += 1

        return X, y
    

class SupervisedDenoisingGenerator(keras.utils.Sequence):
    """
    Outputs (noisy, clean) pairs for validation.
    """

    def __init__(self, 
                 image_dir: str, 
                 noise_model: Callable, 
                 batch_size: int = 16, 
                 patch_edge_size: int = 192, 
                 zero_one_normalisation: bool = True,
                 greyscale: bool = True,
                 **kwargs):

        image_suffixes = ['.jpeg', '.png', '.jpg']
        self.all_image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() 
                                in image_suffixes]
        self.num_images = len(self.all_image_paths)
        if self.num_images == 0:
            raise ValueError(f"The given directory {image_dir} does not contain any images.")
        
        self.val_noise_model = noise_model
        self.batch_size = batch_size
        self.patch_edge_size = patch_edge_size 
        self.zero_one_normalisation = zero_one_normalisation
        self.greyscale = greyscale

        super().__init__(**kwargs)

    def __len__(self):
        return self.num_images // self.batch_size 

    def __getitem__(self, idx):
        batch_size = self.batch_size
        edge_size = self.patch_edge_size

        channels = 1 if self.greyscale else 3
        X = np.zeros((batch_size, edge_size, edge_size, channels), dtype=np.float32)
        y = np.zeros((batch_size, edge_size, edge_size, channels), dtype=np.float32)

        image_counter = 0
        while image_counter < batch_size:
            image_path = random.choice(self.all_image_paths)
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            if self.greyscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=-1)  # Add channel dimension 

            # Normalise if requested
            if self.zero_one_normalisation:
                image = image.astype(np.float32) / 255.0

            h, w, _ = image.shape
            if h < edge_size or w < edge_size:
                continue

            # Randomly choose a patch
            i = np.random.randint(0, h - edge_size + 1)
            j = np.random.randint(0, w - edge_size + 1)
            clean_patch = image[i:i + edge_size, j:j + edge_size]

            # Add noise 
            noisy_patch = self.val_noise_model(clean_patch.astype(np.float32))
            clean_patch = clean_patch.astype(np.float32)

            X[image_counter] = noisy_patch
            y[image_counter] = clean_patch

            image_counter += 1

        return X, y


class SyntheticSpectrogramGenerator(keras.utils.Sequence):
    def __init__(self, 
                 batch_size, num_batches, fs, duration, window, noverlap, 
                 nfft, **kwargs):
        """
        Returns (noisy, clean) synthetic spectrograms for evaluation.
        """

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
        X = np.zeros((self.batch_size, 192, 192, 1)) 
        y = np.zeros((self.batch_size, 192, 192, 1)) 

        image_counter = 0
        for _ in range(self.batch_size):
            _, clean_signal, _ = generate_sine_wave(self.duration, self.fs)
            noisy_signal, _ = add_gaussian_noise(clean_signal)

            # Generate spectrograms
            _, _, noisy_spectrogram = create_spectrogram(noisy_signal, 
                                                           self.fs, self.window, 
                                                           self.noverlap, 
                                                           self.nfft, power=True)
            _, _, clean_spectrogram = create_spectrogram(clean_signal, 
                                                           self.fs, self.window, 
                                                           self.noverlap, 
                                                           self.nfft, power=True)

            # Resize for U-Net dimensions
            noisy_spectrogram = cv2.resize(noisy_spectrogram, (192, 192))
            clean_spectrogram = cv2.resize(clean_spectrogram, (192, 192))

            noisy_spectrogram = np.expand_dims(noisy_spectrogram, axis=-1)
            clean_spectrogram = np.expand_dims(clean_spectrogram, axis=-1)

            # Normalise
            noisy_min = noisy_spectrogram.min(axis=(0, 1), keepdims=True)
            noisy_max = noisy_spectrogram.max(axis=(0, 1), keepdims=True)
            noisy_spectrogram = (noisy_spectrogram - noisy_min) / (noisy_max - noisy_min)

            clean_min = clean_spectrogram.min(axis=(0, 1), keepdims=True)
            clean_max = clean_spectrogram.max(axis=(0, 1), keepdims=True)
            clean_spectrogram = (clean_spectrogram - clean_min) / (clean_max - clean_min)

            X[image_counter] = noisy_spectrogram 
            y[image_counter] = clean_spectrogram 

            image_counter += 1

        return X, y


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
