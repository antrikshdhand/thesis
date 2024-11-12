import numpy as np
import pandas as pd
import keras
import scipy.io
import os
from typing import Literal, Optional, List

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

def import_spectrogram(df: pd.DataFrame, ext: Literal['csv', 'npz', 'mat'], 
                       mat_var_name=None, source_col: str = 'files',
                       dest_col: str = 'spectrogram'):
    """
    Loads spectrogram data from `source_col` column of a DataFrame and 
    attaches it as a new column `dest_col`. The function supports loading 
    data from `.npz`, `.csv`, and `.mat` file formats.

    :param pd.DataFrame df: A DataFrame containing a column `source_col` 
        with file paths to the spectrogram data.
    :param str ext: The file extension of the spectrogram data ('csv', 'npz', or 'mat').
    :param str mat_var_name: The variable name within the `.mat` files to load. 
        This is required when loading MATLAB `.mat` files.
    :param str source_col: The name of the column containing the file paths.
    :param str dest_col: The name of the column to store the spectrograms.
    :return pd.DataFrame: The updated DataFrame with `dest_col` column 
        containing the loaded data, and with the `source_col` column removed.
    :raises ValueError: If `mat_var_name` is not provided for `.mat` files or if 
        an unsupported file extension is used.
    """
    df = df.copy()

    if ext == 'npz':
        df.loc[:, dest_col] = df[source_col].apply(
            lambda x: np.load(x)['np_data']
        )
    elif ext == 'csv':
        df.loc[:, dest_col] = df[source_col].apply(
            lambda x: pd.read_csv(x, header=None).values
        )
    elif ext == 'mat':
        if not mat_var_name:
            raise ValueError("Must provide MAT variable name if ext=mat.")
        df.loc[:, dest_col] = df.loc[:, source_col].apply(
            lambda x: scipy.io.loadmat(x)[mat_var_name]
        )
    else:
        raise ValueError("Unsupported file extension. Use 'npz', 'csv', or 'mat'.")
    
    df = df.drop(columns=[source_col])

    return df

def import_spectrograms(fold_dfs: list[pd.DataFrame], 
                        ext: Literal['csv', 'npz', 'mat'], mat_var_name=None,
                        source_col: str = 'files', dest_col: str = 'spectrogram'):
    """
    Imports spectrograms for each fold in `fold_dfs`. Supports loading from `.npz`, 
    `.csv`, and `.mat` file formats.

    :param list[pd.DataFrame] fold_dfs: A list of DataFrames, each representing 
        a fold to be used in cross-validation. Each DataFrame should contain a 
        column `source_col` with file paths to the spectrogram data.
    :param str ext: The file extension of the spectrogram data ('csv', 'npz', or 'mat').
    :param str mat_var_name: The variable name within the `.mat` files to load.
        This is required when loading MATLAB files.
    :param str source_col: The name of the column containing the file paths.
    :param str dest_col: The name of the column to store the spectrograms.
    :return list[pd.DataFrame]: The updated list of DataFrames, each containing 
        a `spectrogram` column with the loaded data, and without the `files` column.
    """

    for i, fold_df in enumerate(fold_dfs):
        fold_dfs[i] = import_spectrogram(fold_df, ext, mat_var_name, source_col, dest_col)
    return fold_dfs


def generate_kth_fold(fold_dfs: list[pd.DataFrame], test_idx: int, val_idx=None):
    """
    From a list of k dataframes (representing k folds), this function will set 
    aside one fold (index `test_idx`) to be used for testing, another fold 
    (index `val_idx`) to be used for validation (optional), and concatenate 
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
        train_dfs = [fold_dfs[i] for i in range(len(fold_dfs)) if i != test_idx and i != val_idx]
    else:
        # Train DataFrames are all remaining folds excluding the test fold
        train_dfs = [fold_dfs[i] for i in range(len(fold_dfs)) if i != test_idx]

    train_df = pd.concat(train_dfs, ignore_index=True)

    if val_idx is not None:
        return train_df, val_df, test_df
    else:
        return train_df, test_df