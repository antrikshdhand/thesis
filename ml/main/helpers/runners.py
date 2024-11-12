import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import gc
from typing import Literal, List, Dict, Any, Optional, Tuple

from helpers import data
from helpers.data import DeepShipGenerator

def k_fold_cross_validation(
    fold_dfs: List[pd.DataFrame],
    model: keras.Model,
    fit_kwargs: Dict[str, Any],
    eval_kwargs: Dict[str, Any],
    ext: Literal['csv', 'npz', 'mat'],
    mat_var_name: Optional[str] = None,
    data_batch_size: int = 32,
    use_cpu: bool = False
) -> Tuple[keras.Model, List[keras.callbacks.History], List[float], Dict[str, float]]:
    """
    :param List[pd.DataFrame] fold_dfs: A list of DataFrames, each representing 
        one fold.
    :param keras.Model model: A compiled `keras.Model` object.
    :param Dict[str, Any] fit_kwargs: Additional keyword arguments to be fed into 
        `model.fit()`.
    :param Dict[str, Any] eval_kwargs: Additional keyword arguments to be fed into 
        `model.evaluate()`
    :param Literal['csv', 'npz', 'mat] ext: The file extension of the 
        spectrogram data (.csv, .npz, .mat)
    :param Optional[str] mat_var_name: The variable name within the `.mat` files 
        to load if `ext` is 'mat'.
    :param int data_batch_size: Batch size to use with data generators.
    :param bool use_cpu: Toggles training to be run on the CPU instead of GPU.
    :return: A tuple containing the trained model, a list of 
        training histories (one for each fold), evaluation scores 
        (list of loss/metrics for each fold), and a dictionary with averaged
        precision, recall, F1 score, and accuracy across all folds.
    """ 
    histories = []
    evals = []
    
    precisions, recalls, f1_scores, accuracies = [], [], [], []

    NUM_FOLDS = len(fold_dfs)
    for k in range(NUM_FOLDS):
        print(f"FOLD {k + 1}")

        gc.collect()
        keras.backend.clear_session()

        # Set validation fold
        val_idx = 0 if k == NUM_FOLDS - 1 else k + 1

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
        precisions.append(precision_score(y_true, y_pred, average='weighted'))
        recalls.append(recall_score(y_true, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_true, y_pred, average='weighted'))
        accuracies.append(accuracy_score(y_true, y_pred))

        # ---- Clean up ---- #
        del train_df, test_df, train_gen, val_gen, test_gen
        gc.collect()

    # Calculate average metrics across all folds
    metrics = {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1_score": np.mean(f1_scores),
        "accuracy": np.mean(accuracies)
    }

    return model, histories, evals, metrics