import pandas as pd
import numpy as np
import keras

import os
from typing import Literal, Optional, List

from helpers.data import import_spectrogram


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
        :param X_only: If True, outputs (X, X) for autoencoder tasks; otherwise, (X, y).
        """

        self.df = df
        self.ext = ext
        self.mat_var_name = mat_var_name
        self.classes = classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.conv_channel = conv_channel
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
                 conv_channel: bool = True):    
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
        batch_df = import_spectrogram(batch_df, ext='mat', mat_var_name='Pexp',
                                      source_col='file_path_1', dest_col='spec_1')
        # Then convert y spectrogram
        batch_df = import_spectrogram(batch_df, ext='mat', mat_var_name='Pexp',
                                      source_col='file_path_2', dest_col='spec_2')
        
        X = np.stack(batch_df['spec_1'].to_numpy(copy=True))
        y = np.stack(batch_df['spec_2'].to_numpy(copy=True))

        X = np.expand_dims(X, axis=-1)
        y = np.expand_dims(y, axis=-1)

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