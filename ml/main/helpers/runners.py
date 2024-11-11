import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import gc
from typing import Literal

from helpers import data
from helpers.data import DeepShipGenerator

def data_generator(fold_dfs, test_idx, ext: Literal['csv', 'npz', 'mat'],
                    conv_channel=False, verbose=False):

    train_df, test_df = data.generate_kth_fold(fold_dfs, test_idx)
    X_train, X_test, y_train, y_test = data.get_train_test_split(
        train_df, 
        test_df, 
        conv_channel=conv_channel,
        verbose=verbose
    )

    yield X_train, X_test, y_train, y_test

def k_fold_cross_validation_basic_generator(
        fold_dfs: list[pd.DataFrame], model: keras.Model, fit_kwargs: dict, 
        eval_kwargs: dict, use_cpu=False):
    """
    :param list[dict] k_folds: A list of k dictionaries (each representing one
        fold) containing each fold's X_train, X_test, y_train, y_test.
    :param keras.Model model: A compiled `keras.Model` object.
    :param dict fit_kwargs: The remaining keyword arguments to be fed into 
        `model.fit()`.
    :param dict eval_kwargs: Same as `fit_kwargs` but for `model.evaluate()`.
    :param bool use_cpu: Toggles training to be run on the CPU instead of GPU.
        Useful for debugging purposes as sometimes GPU error messages aren't
        clear.

    :return model: The trained model.
    :return histories: A list of length k 
        containing `keras.callbacks.History` objects.
    :return evals: Scalar test loss (if the model has a single output and no
        metrics) or list of scalars (if the model has multiple outputs and/or
        metrics). See `keras.Model.evaluate()` documentation for more info.
    :return metrics: A dictionary containing precision, recall, and F1 score
        averaged across all folds.
    """
    histories = []
    evals = []
    precisions = []
    recalls = []
    f1_scores = []

    # Load spectrograms if using basic generator
    fold_dfs = data.import_spectrograms(fold_dfs, ext='mat', mat_var_name='Pexp')
    
    for k in range(len(fold_dfs)):
        print(f"FOLD {k + 1}")

        gc.collect()
        keras.backend.clear_session()

        X_train, X_test, y_train, y_test = next(data_generator(
            fold_dfs=fold_dfs,
            test_idx=k,
            ext="mat",
            conv_channel=True,
            verbose=False
        ))

        # ---- Model fitting ---- #
        if use_cpu:
            with tf.device('/CPU:0'):
                history = model.fit(
                    x=X_train,
                    y=y_train,
                    **fit_kwargs
                )
        else:
            history = model.fit(
                x=X_train,
                y=y_train,
                **fit_kwargs
            )

        histories.append(history)

        # ---- Model evaluation ---- #
        evaluation = model.evaluate(
            x=X_test,
            y=y_test,
            **eval_kwargs
        ) 
        evals.append(evaluation)

        # ---- Predictions and Metrics ---- #
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)  # Get class predictions
        y_true = np.argmax(y_test, axis=1)

        # Calculate precision, recall, and F1 score for the current fold
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # ---- Clean up ---- #
        del X_train, X_test, y_train, y_test
        gc.collect()

    # Calculate average precision, recall, and F1 score across all folds
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1_score = np.mean(f1_scores)

    metrics = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1_score
    }
    
    return model, histories, evals, metrics

def k_fold_cross_validation(
        fold_dfs: list[pd.DataFrame], model: keras.Model, fit_kwargs: dict, 
        eval_kwargs: dict, ext: Literal['csv', 'npz', 'mat'], mat_var_name=None,
        data_batch_size=32, use_cpu=False):
    """
    :param list[pd.DataFrame] fold_dfs: A list of DataFrames, each representing 
        one fold.
    :param keras.Model model: A compiled `keras.Model` object.
    :param dict fit_kwargs: The remaining keyword arguments to be fed into 
        `model.fit()`.
    :param dict eval_kwargs: Same as `fit_kwargs` but for `model.evaluate()`.
    :param str ext: The file extension of the spectrogram data (.csv, .npz, .mat)
    :param str mat_var_name: The variable name within the `.mat` files to load 
        if `ext` is 'mat'.
    :param int data_batch_size: Batch size to use with data generators.
    :param bool use_cpu: Toggles training to be run on the CPU instead of GPU.
    :return: Tuple containing trained model, list of histories, evaluation 
        scores, and metrics dictionary.
    """ 
    histories = []
    evals = []
    
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    NUM_FOLDS = len(fold_dfs)
    for k in range(NUM_FOLDS):
        print(f"FOLD {k + 1}")

        gc.collect()
        keras.backend.clear_session()

        # Load data for current fold
        if k == NUM_FOLDS - 1:
            val_idx = 0
        else:
            val_idx = k + 1

        train_df, val_df, test_df = data.generate_kth_fold(
            fold_dfs, 
            test_idx=k, 
            val_idx=val_idx
        )

        train_gen = DeepShipGenerator(train_df, ext=ext, mat_var_name=mat_var_name, 
                                      batch_size=data_batch_size, 
                                      conv_channel=True)
        test_gen = DeepShipGenerator(test_df, ext=ext, mat_var_name=mat_var_name, 
                                     batch_size=data_batch_size, 
                                     conv_channel=True, shuffle=False)
        val_gen = DeepShipGenerator(val_df, ext=ext, mat_var_name=mat_var_name,
                                        batch_size=data_batch_size,
                                        conv_channel=True, shuffle=False)

        # ---- Model fitting ---- #
        if use_cpu:
            with tf.device('/CPU:0'):
                history = model.fit(
                    train_gen,
                    validation_data=val_gen,
                    **fit_kwargs
                )
        else:
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                **fit_kwargs
            )

        histories.append(history)

        # # ---- Model evaluation ---- #
        evaluation = model.evaluate(
            test_gen,
            **eval_kwargs
        ) 
        evals.append(evaluation)

        # # ---- Predictions and Metrics ---- #
        y_true = []
        y_pred = []
        for batch_x, batch_y in test_gen:
            preds = model.predict(batch_x)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(batch_y, axis=1))

        # Calculate precision, recall, and F1 score for the current fold
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)

        # ---- Clean up ---- #
        del train_df, test_df, train_gen, val_gen, test_gen
        gc.collect()

    # Calculate average precision, recall, and F1 score across all folds
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1_score = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracies)

    metrics = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1_score,
        "accuracy": avg_accuracy
    }
    
    return model, histories, evals, metrics