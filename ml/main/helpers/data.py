import numpy as np
import pandas as pd
import keras
import scipy.io
from typing import Literal

def rename_folds(fold_definitions: pd.DataFrame, new_path_to_root: str, 
                  unix: bool, ext: Literal['csv', 'npz', 'mat'],
                  label_encodings: dict):
    """
    Replaces paths in the original fold definition csv with user's local setup.
    Replaces .wav with .csv, .npz or .mat depending on `ext`. 

    :param pd.DataFrame fold_definitions: DataFrame of fold definitions.
    :param str new_path_to_root: Where the root folder containing the class
        subfolders exists on the user's machine, without any preceding or
        trailing slashes.
    :param bool unix: Whether to conform to unix-style paths or not.
    :param str format: What format the files are saved as. One of ['csv', 'npz',
        'mat']
    :param dict label_encodings: A dictionary defining the mapping of 
        integer encodings used in the fold definition file to class labels.
    :return pd.DataFrame fold_definitions: The updated DataFrame of fold 
        definitions with the `files` column renamed to the current user setup.
    """

    fold_definitions = fold_definitions.copy()
    
    # Points to the folder that contains the class subfolders in the csv file.
    first_entry_label = fold_definitions["labels"].iloc[0]
    prev_path_to_root = fold_definitions.iloc[0, 0].split(f"/{label_encodings[first_entry_label]}")[0]

    fold_definitions['files'] = fold_definitions['files'].apply(
        lambda x: x.replace(prev_path_to_root, new_path_to_root)
    )

    # x86 directory formatting
    if not unix:
        fold_definitions['files'] = fold_definitions['files'].apply(
            lambda x: x.replace('/', '\\')
        )
    
    # Convert .wav to .csv or .npz or .mat
    if ext == 'csv':
        fold_definitions['files'] = fold_definitions['files'].apply(
            lambda x: x.replace('.wav', '.csv')
        )
    elif ext == 'npz':
        fold_definitions['files'] = fold_definitions['files'].apply(
            lambda x: x.replace('.wav', '.npz') 
        )
    elif ext == 'mat':
        fold_definitions['files'] = fold_definitions['files'].apply(
            lambda x: x.replace('.wav', '.mat')
        )

    return fold_definitions


def get_fold_dfs(fold_definition_csv: str, new_path_to_root: str,
                ext: Literal['csv', 'npz', 'mat'], n_folds=None, unix=False,
                label_encodings={0:'Tanker', 1:'Cargo', 2:'Tug', 3:'Passengership'}):
    """
    Make one-hot encoded folds based on fold definition csv and `n_folds`. Updates
    the `files` column to point to `new_path_to_root`.

    :param str fold_definition_csv: Path to the fold definition csv.
    :param str new_path_to_root: Where the root folder containing the class
        subfolders (which contain files of spectrograms) exists on the 
        user's machine, without any preceding or trailing slashes. 
    :param str ext: What type of files are contained in `new_path_to_root`.
        One of ['csv', 'npz', 'mat'].
    :param dict label_encodings: A dictionary defining the mapping of 
        integer encodings used in the fold definition file to class labels.
    :param int n_folds: The number of folds you wish to limit the dataset to.
        Should be an integer between 2 and the true number of folds given in the
        fold definition csv. If `None` then defaults to the maximum number of
        folds.
    :return fold_dfs: A list of k dataframes (k folds) with columns 
        [class_1, class_2, ..., class_n, files].
    :return total_samples: The total number of data points across all folds 
        in the returned dataset.
    """
    fold_definitions = pd.read_csv(fold_definition_csv)
    fold_definitions = rename_folds(
        fold_definitions, 
        new_path_to_root,
        unix=unix,
        ext=ext,
        label_encodings=label_encodings
    )

    actual_n_folds = max(fold_definitions['folds']) + 1
    if not n_folds:
        N_FOLDS = actual_n_folds
    else:
        if n_folds < 2 or n_folds > actual_n_folds:
            raise IndexError('n_folds should be an integer between 2 and the max number of folds in the given csv')
        else:
            N_FOLDS = n_folds

    
    # List holding one pd.DataFrame for each fold.
    fold_dfs = [
        fold_definitions.query('folds == @i').copy().reset_index(drop=True) 
            for i in range(N_FOLDS)
    ]

    for i, fold_df in enumerate(fold_dfs):
        fold_df = fold_df.drop(columns=["folds"])

        # One-hot encode classes.
        fold_df['labels'] = fold_df['labels'].replace(label_encodings)
        fold_df = pd.get_dummies(
            fold_df, columns=['labels'], prefix='', prefix_sep='', dtype=int
        )

        fold_dfs[i] = fold_df

    return fold_dfs


def import_spectrogram(fold_df: pd.DataFrame, 
                        ext: Literal['csv', 'npz', 'mat'], mat_var_name=None):
    """
    Loads spectrogram data from the `files` column of a single fold DataFrame and 
    attaches it as a new column `spectrogram`. The function supports loading 
    data from `.npz`, `.csv`, and `.mat` file formats.

    :param pd.DataFrame fold_df: A DataFrame containing a column `files` 
        with file paths to the spectrogram data for a particular fold.
    :param str ext: The file extension of the spectrogram data ('csv', 'npz', or 'mat').
    :param str mat_var_name: The variable name within the `.mat` files to load 
        if `ext` is 'mat'. This is required when loading MATLAB `.mat` files.
    :return pd.DataFrame: The updated DataFrame with a `spectrogram` column 
        containing the loaded data, and with the `files` column removed.
    :raises ValueError: If `mat_var_name` is not provided for `.mat` files or if 
        an unsupported file extension is used.
    """
    fold_df = fold_df.copy()
    if ext == 'npz':
        fold_df.loc[:, 'spectrogram'] = fold_df['files'].apply(
            lambda x: np.load(x)['np_data']
        )
    elif ext == 'csv':
        fold_df.loc[:, 'spectrogram'] = fold_df['files'].apply(
            lambda x: pd.read_csv(x, header=None).values
        )
    elif ext == 'mat':
        if not mat_var_name:
            raise ValueError("Must provide MAT variable name if ext=mat.")
        fold_df.loc[:, 'spectrogram'] = fold_df.loc[:, 'files'].apply(
            lambda x: scipy.io.loadmat(x)[mat_var_name]
        )
    else:
        raise ValueError("Unsupported file extension. Use 'npz', 'csv', or 'mat'.")
    
    fold_df = fold_df.drop(columns=['files'])

    return fold_df

def import_spectrograms(fold_dfs: list[pd.DataFrame], 
                        ext: Literal['csv', 'npz', 'mat'], mat_var_name=None):
    """
    Loads spectrogram data from the `files` column of each fold in `fold_dfs` and 
    attaches it as a new column `spectrogram`. Supports loading from `.npz`, 
    `.csv`, and `.mat` file formats.

    :param list[pd.DataFrame] fold_dfs: A list of DataFrames, each representing 
        a fold to be used in cross-validation. Each DataFrame should contain a 
        column `files` with file paths to the spectrogram data.
    :param str ext: The file extension of the spectrogram data ('csv', 'npz', or 'mat').
    :param str mat_var_name: The variable name within the `.mat` files to load 
        if `ext` is 'mat'. This is required when loading MATLAB `.mat` files.
    :return list[pd.DataFrame]: The updated list of DataFrames, each containing 
        a `spectrogram` column with the loaded data, and without the `files` column.
    """

    for i, fold_df in enumerate(fold_dfs):
        fold_dfs[i] = import_spectrogram(fold_df, ext, mat_var_name)
    return fold_dfs


def generate_kth_fold(fold_dfs: list[pd.DataFrame], test_idx: int, val_idx=None):
    """
    From a list of k dataframes (representing k folds), this function will set 
    aside one fold (index `test_idx`) to be used for testing, another fold 
    (index `val_idx`, if provided) to be used for validation, and concatenate 
    all remaining folds to be used for training.

    :param fold_dfs: List of DataFrames, each representing a fold.
    :param test_idx: Index of the fold to be used as the test set.
    :param val_idx: Optional index of the fold to be used as the validation set.
    :return: If `val_idx` is provided, returns a tuple (train_df, val_df, test_df).
             Otherwise, returns a tuple (train_df, test_df).
    """
    if test_idx >= len(fold_dfs) or test_idx < 0:
        raise IndexError("Test fold index must be within the range of folds")

    if val_idx is not None:
        if val_idx >= len(fold_dfs) or val_idx < 0:
            raise IndexError("Validation fold index must be within the range of folds")
        if val_idx == test_idx:
            raise ValueError("Validation fold index cannot be the same as test fold index")

    # Separate out the test DataFrame
    test_df = fold_dfs[test_idx]

    # Separate out the validation DataFrame if specified
    if val_idx is not None:
        val_df = fold_dfs[val_idx]
        # Train DataFrames are all remaining folds excluding the test and validation folds
        train_dfs = [
            fold_dfs[i] for i in range(len(fold_dfs)) 
            if i != test_idx and i != val_idx
        ]
    else:
        # Train DataFrames are all remaining folds excluding the test fold
        train_dfs = [
            fold_dfs[i] for i in range(len(fold_dfs)) if i != test_idx
        ]

    train_df = pd.concat(train_dfs, ignore_index=True)

    if val_idx is not None:
        return train_df, val_df, test_df
    else:
        return train_df, test_df


def get_train_test_split(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         conv_channel: bool, verbose=False,
                         classes=['Cargo', 'Passengership', 'Tanker', 'Tug']):
    """
    Given one dataframe containing all the data points to be used for training,
    and one dataframe containing all the data points to be used for testing,
    this function will return the X_train, X_test, y_train, y_test arrays to be
    used for model fitting and validation.

    The dataframes should have columns:
    ['class1', 'class2', ..., 'classn', 'spectrogram'].

    If `conv == true` then a channel dimension is added to `X_train` and
    `X_test` to be used with convolution operations.
    """

    X_train = train_df['spectrogram'].to_numpy(copy=True)
    X_test = test_df['spectrogram'].to_numpy(copy=True)

    X_train = np.stack(X_train)
    X_test = np.stack(X_test)

    if conv_channel: # Add channel dim for CNN
        # X_train and X_test should have 4 dimensions:
        # (number of samples, number of time bins, number of freqs, channel)
        X_train = np.expand_dims(X_train, axis=-1) 
        X_test = np.expand_dims(X_test, axis=-1)

    y_train = train_df[classes].to_numpy(copy=True)
    y_test = test_df[classes].to_numpy(copy=True)

    if verbose:
        print()
        print("Data dimensions:")
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)

    return X_train, X_test, y_train, y_test


class DeepShipGenerator(keras.utils.Sequence):

    def __init__(self, df, ext, mat_var_name=None,
                 classes=['Cargo', 'Passengership', 'Tanker', 'Tug'],
                 batch_size=8, input_size=(192, 192, 1), shuffle=True,
                 conv_channel=True, X_only=False):
        """
        Generator for loading spectrogram data in batches.
        
        :param df: DataFrame containing file paths and labels.
        :param ext: File extension (csv, npz, mat) for spectrogram files.
        :param mat_var_name: Variable name in .mat files if applicable.
        :param classes: List of class labels (for one-hot encoding).
        :param batch_size: Size of each batch.
        :param input_size: Expected input dimensions (height, width, channels).
        :param shuffle: Whether to shuffle the data at the end of each epoch.
        :param conv_channel: Whether to add a channel dimension for CNN input.
        :param X_only: Whether to output tuples of (X, X), useful for autoencoders.
        """
        self.df = df
        self.ext = ext
        self.mat_var_name = mat_var_name
        self.classes = classes
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.conv_channel = conv_channel
        self.X_only = X_only

        self.n = len(self.df)
        self.n_classes = len(classes)
        self.y_col = classes  
        
        # Shuffle the data initially
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        """
        Number of batches per epoch.
        """
        return self.n // self.batch_size

    def __get_x(self, batch_df):
        """
        Load spectrograms for the batch.
        """
        # Import spectrograms for each file in the batch
        batch_df = import_spectrogram(batch_df, self.ext, self.mat_var_name)
        X = np.stack(batch_df['spectrogram'].to_numpy(copy=True))

        # Add channel dimension if using convolutional layers
        if self.conv_channel:
            X = np.expand_dims(X, axis=-1)

        return X

    def __get_y(self, batch_df):
        """
        Get one-hot encoded labels for the batch.
        """
        return batch_df[self.y_col].to_numpy(copy=True)

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
        Shuffle the data at the end of each epoch.
        """
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)


# def get_splits_for_all_folds(fold_dfs: list[pd.DataFrame], conv_channel=False,
#                              verbose=False):
#     """
#     Returns a list of k dictionaries each containing 
#     `X_train`, `X_test`, `y_train`, and `y_test` for the k-th fold.
#     """
#     k_folds = []

#     for k in range(len(fold_dfs)):
#         if verbose:
#             print(f'\nProcessing fold: {k + 1}')

#         kth_dict = {}

#         train_df, test_df = generate_kth_fold(fold_dfs, k)
#         X_train, X_test, y_train, y_test = \
#             get_train_test_split(train_df, test_df, conv_channel=conv_channel, 
#                                  verbose=verbose)

#         kth_dict['X_train'] = X_train
#         kth_dict['X_test'] = X_test
#         kth_dict['y_train'] = y_train
#         kth_dict['y_test'] = y_test

#         k_folds.append(kth_dict)

#     return k_folds

# def normalise_spectrograms(k_folds: list[dict]):
#     """
#     Given a list of k dictionaries, each containing the 
#     `X_train`, `X_test`, `y_train`, and `y_test` for the k-th fold, this
#     function will normalise both the X_train and X_test spectrograms using
#     the global max and min.
#     """

#     maximums = []
#     minimums = []
#     for kth_dict in k_folds:
#         maximums.append(kth_dict['X_train'].max())
#         maximums.append(kth_dict['X_test'].max())

#         minimums.append(kth_dict['X_train'].min())
#         minimums.append(kth_dict['X_test'].min())
    
#     maximum = max(maximums)
#     minimum = min(minimums)

#     for kth_dict in k_folds:
#         kth_dict['X_train'] = (kth_dict['X_train'] - minimum) / (maximum - minimum)
#         kth_dict['X_test'] = (kth_dict['X_test'] - minimum) / (maximum - minimum)
    
#     return k_folds


# ##### tf.data.Dataset #####

# def generator(X, y):
#     for i in range(len(X)):
#         yield X[i], y[i]

# def convert_to_tf_dataset(X, y, batch_size=32):
#     output_signature = (
#         tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32),  # Shape of individual input sample
#         tf.TensorSpec(shape=y.shape[1:], dtype=tf.float32)   # Shape of individual label
#     )
    
#     dataset = tf.data.Dataset.from_generator(
#         lambda: generator(X, y), 
#         output_signature=output_signature
#     )

#     return dataset.batch(batch_size)

# def get_tf_datasets_for_all_folds(fold_dfs: list[pd.DataFrame],
#                                   conv_channel=False, validation_split=None,
#                                   verbose=False):
#     """ 
#     Returns a list of k tuples each containing a `tf.data.Dataset` for training 
#     (index 0), testing (index 1), and optionally for validation (index 2).

#     :param list[pd.DataFrame] fold_dfs: A list of fold dataframes.
#     :param bool conv_channel: Whether to add an extra dimension for convs.
#     :param float validation_split: Percentage of training dataset to use for
#         validation data. If `None`, no validation dataset is created.
#     """
#     k_folds = []
#     for k in range(len(fold_dfs)):
#         if verbose:
#             print(f'\nProcessing fold: {k + 1}')

#         train_df, test_df = generate_kth_fold(fold_dfs, k)
#         X_train, X_test, y_train, y_test = \
#             get_train_test_split(train_df, test_df, conv_channel=conv_channel, 
#                                  verbose=verbose)

#         train_dataset = convert_to_tf_dataset(X_train, y_train)
#         test_dataset = convert_to_tf_dataset(X_test, y_test)

#         if validation_split:
#             if validation_split < 0 or validation_split > 1:
#                 raise Exception('Validation split should be between 0 and 1') 
            
#             train_dataset, validation_dataset = tf.keras.utils.split_dataset(
#                 train_dataset,
#                 right_size=validation_split,
#                 shuffle=False
#             )
            
#             k_folds.append((train_dataset, test_dataset, validation_dataset))
#         else:
#             k_folds.append((train_dataset, test_dataset))

#     return k_folds