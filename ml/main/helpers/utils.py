import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches
import cv2
import pandas as pd
from typing import Union

from helpers import noise_models

def check_gpu_use():
    if len(tf.config.list_physical_devices('GPU')) == 0:
        raise Exception('****** NO GPU DETECTED! ******')
    print('GPU DETECTED \u2713')
    print(tf.config.list_physical_devices('GPU'))

def calculate_metrics(evals: list):
    mean = np.mean(evals, axis=0)
    return mean[0], mean[1] # loss, acc
    
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim(y_true, y_pred):
  return tf.image.ssim(y_true, y_pred, 1.0)

def get_history_curve(history: keras.callbacks.History, metrics: list[str]):
    """
    Plots the training history for specified metrics over epochs.
    :param history: Keras History object from model.fit()
    :param metrics: List of metric names to plot (e.g., ['loss', 'psnr'])
    :return: matplotlib figure containing the plots
    """
    epochs = np.arange(1, len(history.history[metrics[0]]) + 1, dtype=int)
    num_metrics = len(metrics)

    if num_metrics == 1:
        metric_data = history.history[metrics[0]]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, metric_data, 'bo-')
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_title(f'{metrics[0]} over epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metrics[0])
        ax.grid(True)

    else:
        fig, axes = plt.subplots(num_metrics, 1, figsize=(8, 10), sharex=True)

        for i, metric in enumerate(metrics):
            metric_data = history.history[metric]
            axes[i].plot(epochs, metric_data, 'bo-')
            axes[i].xaxis.get_major_locator().set_params(integer=True)
            axes[i].set_title(f'{metric} over epochs')
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel(metric)
            axes[i].grid(True)

        fig.tight_layout()

    return fig


def _ssim_plot(ax, epochs, history_dict):
    ax.plot(epochs, history_dict['ssim'], label='SSIM', marker='o')
    ax.plot(epochs, history_dict['val_ssim'], label='Validation SSIM', marker='s')
    ax.set_title('SSIM over epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('SSIM')
    ax.legend()
    ax.grid(True)
    return ax

def _loss_plot(ax, epochs, history_dict):
    ax.plot(epochs, history_dict['loss'], label='Loss', marker='o')
    ax.plot(epochs, history_dict['val_loss'], label='Validation loss', marker='s')
    ax.set_title('Loss over epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    return ax

def _psnr_plot(ax, epochs, history_dict):
    ax.plot(epochs, history_dict['psnr'], label='PSNR', marker='o')
    ax.plot(epochs, history_dict['val_psnr'], label='Validation PSNR', marker='s')
    ax.set_title('PSNR over epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('PSNR (dB)')
    ax.legend()
    ax.grid(True)
    return ax

def _psnr_loss_curves_from_dict(history_dict: dict, together=False, plots=['psnr', 'loss', 'ssim']):
    # Find the first valid key in history_dict to determine the number of epochs
    first_key = next((key for key in plots if key in history_dict), None)
    if first_key is None:
        raise ValueError("You can only plot a metric that was used in the model!")
    num_epochs = len(history_dict[first_key])
    epochs = np.arange(1, num_epochs + 1, dtype=int)

    if together:
        if len(plots) == 3:
            fig, (ax_psnr, ax_loss, ax_ssim) = plt.subplots(3, 1, figsize=(6, 11))
            ax_psnr = _psnr_plot(ax_psnr, epochs, history_dict) 
            ax_loss = _loss_plot(ax_loss, epochs, history_dict)
            ax_ssim = _ssim_plot(ax_ssim, epochs, history_dict)
            fig.tight_layout(pad=1.5)
            return fig
        elif len(plots) == 2:
            fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(11, 5)) # (10, 5) is pretty good
            if 'psnr' in plots:
                ax_1 = _psnr_plot(ax_1, epochs, history_dict) 
                if 'loss' in plots:
                    ax_2 = _loss_plot(ax_2, epochs, history_dict)
                else:
                    ax_2 = _ssim_plot(ax_2, epochs, history_dict)
            else:
                ax_1 = _loss_plot(ax_1, epochs, history_dict)
                ax_2 = _ssim_plot(ax_2, epochs, history_dict)
            
            fig.tight_layout()
            return fig
    else:
        created_figures = []
        
        if 'psnr' in plots:
            fig_psnr, ax_psnr = plt.subplots()
            ax_psnr = _psnr_plot(ax_psnr, epochs, history_dict)
            created_figures.append(fig_psnr)

        if 'loss' in plots:
            fig_loss, ax_loss = plt.subplots()
            ax_loss = _loss_plot(ax_loss, epochs, history_dict)
            created_figures.append(fig_loss)

        if 'ssim' in plots:
            fig_ssim, ax_ssim = plt.subplots()
            ax_ssim = _ssim_plot(ax_ssim, epochs, history_dict)
            created_figures.append(fig_ssim)

        return tuple(created_figures)

def get_psnr_loss_ssim_curves(history: Union[keras.callbacks.History, str], together=False, plots=['psnr', 'loss', 'ssim']):
    """
    Generate and return PSNR, Loss, and SSIM curves for model training.

    Args:
    - history (Union[keras.callbacks.History, str]): 
        If a `keras.callbacks.History` object, it should contain the training metrics 
        (e.g., 'psnr', 'loss', 'ssim'). If a string, it should be a filepath to a CSV 
        file containing training metrics with columns for each metric (e.g., 'psnr', 'loss', 'ssim') 
        indexed by epochs.
    - together (bool, optional): 
        If `True`, returns one figure with all requested plots combined. If `False`, 
        returns individual figures for PSNR, Loss, and SSIM. Default is `False`.
    - plots (list, optional): 
        A list of plots to generate. Possible values are ['psnr', 'loss', 'ssim']. 
        Default is `['psnr', 'loss', 'ssim']`.

    Returns:
    - If `together=True`, returns a single figure containing the requested plots.
    - If `together=False`, returns a tuple of individual figures in the order PSNR, Loss, and SSIM.

    Note:
    - The function either takes a `keras.callbacks.History` object or a file path to a CSV containing the training metrics.
    """
    if type(history) == keras.callbacks.History:
        return _psnr_loss_curves_from_dict(history.history, together, plots)
    elif type(history) == str:
        df = pd.read_csv(history, index_col="epoch")
        history_dict = df.to_dict(orient='list')
        return _psnr_loss_curves_from_dict(history_dict, together, plots)


def get_acc_loss_curves_by_epoch(histories: list[keras.callbacks.History], overlay=False):
    N_FOLDS = len(histories)
    max_epochs = max(len(fold.history['acc']) for fold in histories)  # Find the maximum epochs across folds
    epochs = np.arange(1, max_epochs + 1, dtype=int)  # Epochs range from 1 to max epochs

    # Initialize lists for storing accuracies and losses for each fold
    accuracies = []
    val_accuracies = []
    losses = []
    val_losses = []

    for fold in histories:
        n_epochs = len(fold.history['acc'])  # Number of epochs for this fold
        accuracies.append(fold.history['acc'] + [None] * (max_epochs - n_epochs))  # Fill remaining epochs with None
        val_accuracies.append(fold.history['val_acc'] + [None] * (max_epochs - n_epochs))
        losses.append(fold.history['loss'] + [None] * (max_epochs - n_epochs))
        val_losses.append(fold.history['val_loss'] + [None] * (max_epochs - n_epochs))

    accuracies = np.array(accuracies, dtype=object)  # Use object dtype to allow None values
    val_accuracies = np.array(val_accuracies, dtype=object)
    losses = np.array(losses, dtype=object)
    val_losses = np.array(val_losses, dtype=object)

    # Calculate averages for each epoch, ignoring None values
    mean_accuracy = np.nanmean(np.where(accuracies == None, np.nan, accuracies), axis=0)
    mean_val_accuracy = np.nanmean(np.where(val_accuracies == None, np.nan, val_accuracies), axis=0)
    mean_loss = np.nanmean(np.where(losses == None, np.nan, losses), axis=0)
    mean_val_loss = np.nanmean(np.where(val_losses == None, np.nan, val_losses), axis=0)

    ALPHA = 0.15
    if overlay:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(14, 5))

        # Plot each fold
        for fold_i in range(N_FOLDS):
            fold_epochs = np.arange(1, len(histories[fold_i].history['acc']) + 1, dtype=int)
            ax[0].plot(fold_epochs, accuracies[fold_i][:len(fold_epochs)], 'o--b', alpha=ALPHA)
            ax[0].plot(fold_epochs, val_accuracies[fold_i][:len(fold_epochs)], 'o--', color='orange', alpha=ALPHA)
            ax[1].plot(fold_epochs, losses[fold_i][:len(fold_epochs)], 'o--b', alpha=ALPHA)
            ax[1].plot(fold_epochs, val_losses[fold_i][:len(fold_epochs)], 'o--', color='orange', alpha=ALPHA)
        
        # Plot mean
        ax[0].plot(epochs, mean_accuracy, 'D-b', markersize=7, label='Training')
        ax[0].plot(epochs, mean_val_accuracy, 'D-', markersize=7, color='orange', label='Validation')
        ax[1].plot(epochs, mean_loss, 'D-b', markersize=7, label='Training')
        ax[1].plot(epochs, mean_val_loss, 'D-', markersize=7, color='orange', label='Validation')

        ax[0].set_title('Average accuracy over all folds')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_ylim(bottom=0, top=1)

        ax[1].set_title('Average loss over all folds')
        ax[1].set_ylabel('Loss')
        ax[1].set_ylim(bottom=0)

        for j in range(2):
            ax[j].set_xlabel('Epochs')
            ax[j].xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax[j].legend(loc='upper left')
            ax[j].grid()

    else:
        fig, ax = plt.subplots(
            nrows=N_FOLDS,
            ncols=2, 
            sharex='none',
            sharey='col',
            figsize=(14, 5 * N_FOLDS)
        )

        for fold_i in range(N_FOLDS):
            fold_epochs = np.arange(1, len(histories[fold_i].history['acc']) + 1, dtype=int)

            ax[fold_i, 0].plot(fold_epochs, accuracies[fold_i][:len(fold_epochs)], 'o-b', label='Training')
            ax[fold_i, 0].plot(fold_epochs, val_accuracies[fold_i][:len(fold_epochs)], 'o--', color='orange', label='Validation')
            ax[fold_i, 0].set_title(f'Fold {fold_i + 1} - Accuracy')
            ax[fold_i, 0].set_ylabel('Accuracy')
            ax[fold_i, 0].set_ylim(bottom=0, top=1)

            ax[fold_i, 1].plot(fold_epochs, losses[fold_i][:len(fold_epochs)], 'o-b', label='Training')
            ax[fold_i, 1].plot(fold_epochs, val_losses[fold_i][:len(fold_epochs)], 'o--', color='orange', label='Validation')
            ax[fold_i, 1].set_title(f'Fold {fold_i + 1} - Loss')
            ax[fold_i, 1].set_ylabel('Loss')

            # Features common to both columns of subplots
            for j in range(2):
                ax[fold_i, j].set_xlabel('Epochs')
                ax[fold_i, j].legend(loc='upper left')
                ax[fold_i, j].xaxis.set_major_locator(mticker.MultipleLocator(1))
                ax[fold_i, j].grid()

        fig.suptitle('Training Accuracy and Loss', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        
    return fig


def get_acc_loss_curves_by_fold(histories: list[keras.callbacks.History], overlay=False):
    N_FOLDS = len(histories)
    max_epochs = max(len(fold.history['acc']) for fold in histories)
    folds = np.arange(1, N_FOLDS + 1, dtype=int)

    # Initialize lists for storing accuracies and losses for each epoch
    epoch_accuracies = []
    epoch_val_accuracies = []
    epoch_losses = []
    epoch_val_losses = []

    for epoch in range(max_epochs):
        fold_accuracies = []
        fold_val_accuracies = []
        fold_losses = []
        fold_val_losses = []

        for fold in histories:
            if epoch < len(fold.history['acc']):
                fold_accuracies.append(fold.history['acc'][epoch])
                fold_val_accuracies.append(fold.history['val_acc'][epoch])
                fold_losses.append(fold.history['loss'][epoch])
                fold_val_losses.append(fold.history['val_loss'][epoch])
            else:
                fold_accuracies.append(None)
                fold_val_accuracies.append(None)
                fold_losses.append(None)
                fold_val_losses.append(None)

        epoch_accuracies.append(fold_accuracies)
        epoch_val_accuracies.append(fold_val_accuracies)
        epoch_losses.append(fold_losses)
        epoch_val_losses.append(fold_val_losses)

    epoch_accuracies = np.array(epoch_accuracies, dtype=object)
    epoch_val_accuracies = np.array(epoch_val_accuracies, dtype=object)
    epoch_losses = np.array(epoch_losses, dtype=object)
    epoch_val_losses = np.array(epoch_val_losses, dtype=object)

    ALPHA = 0.15
    if overlay:
        if max_epochs == 1:
            print("WARNING: plot is identical regardless of overlay setting for epochs=1")

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(14, 5))

        # Plot each epoch
        for epoch_i in range(max_epochs):
            ax[0].plot(folds, epoch_accuracies[epoch_i], 'o--b', alpha=ALPHA)
            ax[0].plot(folds, epoch_val_accuracies[epoch_i], 'o--', color='orange', alpha=ALPHA)

            ax[1].plot(folds, epoch_losses[epoch_i], 'o--b', alpha=ALPHA)
            ax[1].plot(folds, epoch_val_losses[epoch_i], 'o--', color='orange', alpha=ALPHA)

        # Plot mean across epochs for each fold
        mean_accuracy_per_fold = np.nanmean(epoch_accuracies, axis=0)
        mean_val_accuracy_per_fold = np.nanmean(epoch_val_accuracies, axis=0)
        mean_loss_per_fold = np.nanmean(epoch_losses, axis=0)
        mean_val_loss_per_fold = np.nanmean(epoch_val_losses, axis=0)

        ax[0].plot(folds, mean_accuracy_per_fold, 'D-b', markersize=7, label='Training')
        ax[0].plot(folds, mean_val_accuracy_per_fold, 'D-', color='orange', markersize=7, label='Validation')
        ax[1].plot(folds, mean_loss_per_fold, 'D-b', markersize=7, label='Training')
        ax[1].plot(folds, mean_val_loss_per_fold, 'D-', color='orange', markersize=7, label='Validation')

        ax[0].set_title('Average accuracy over all epochs')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_ylim(bottom=0, top=1)

        ax[1].set_title('Average loss over all epochs')
        ax[1].set_ylabel('Loss')
        ax[1].set_ylim(bottom=0)

        for j in range(2):
            ax[j].set_xlabel('Folds')
            ax[j].xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax[j].legend(loc='upper left')
            ax[j].grid()

    else:
        
        # Handle case where there is only one epoch
        if max_epochs == 1:
            print("WARNING: plot is identical regardless of overlay setting for epochs=1")
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
            ax = np.expand_dims(ax, axis=0)  # Add extra dimension to keep consistent indexing

        else:
            fig, ax = plt.subplots(
                nrows=max_epochs,
                ncols=2, 
                sharex='none',
                sharey='col',
                figsize=(14, 5 * max_epochs)
            )

        for epoch_i in range(max_epochs):
            ax[epoch_i, 0].plot(folds, epoch_accuracies[epoch_i], 'o-b', label='Training')
            ax[epoch_i, 0].plot(folds, epoch_val_accuracies[epoch_i], 'o--', color='orange', label='Validation')
            ax[epoch_i, 0].set_title(f'Epoch {epoch_i + 1} - Accuracy')
            ax[epoch_i, 0].set_ylabel('Accuracy')
            ax[epoch_i, 0].set_ylim(bottom=0, top=1)

            ax[epoch_i, 1].plot(folds, epoch_losses[epoch_i], 'o-b', label='Training')
            ax[epoch_i, 1].plot(folds, epoch_val_losses[epoch_i], 'o--', color='orange', label='Validation')
            ax[epoch_i, 1].set_title(f'Epoch {epoch_i + 1} - Loss')
            ax[epoch_i, 1].set_ylabel('Loss')

            for j in range(2):
                ax[epoch_i, j].set_xlabel('Folds')
                ax[epoch_i, j].legend(loc='upper left')
                ax[epoch_i, j].xaxis.set_major_locator(mticker.MultipleLocator(1))
                ax[epoch_i, j].grid()

        fig.suptitle('Training Accuracy and Loss by Fold', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

    return fig

############ DENOISING ############

def _image_to_ground_truth_patch(image_path, patch_coords=None, patch_size=192,
                                zero_one_normalisation=True, greyscale=True):

    if greyscale:
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_image = np.expand_dims(original_image, axis=-1)
    else:
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if zero_one_normalisation:
        original_image = original_image.astype(np.float32) / 255.0

    h, w, _ = original_image.shape
    if h < patch_size or w < patch_size:
        raise ValueError(f"Image size ({h}, {w}) is smaller than patch size {patch_size}.")

    if patch_coords is None:
        # Select a random patch
        top_left_w = np.random.randint(0, w - patch_size + 1)
        top_left_h = np.random.randint(0, h - patch_size + 1)
    else:
        top_left_w = patch_coords[0]
        top_left_h = patch_coords[1]
    top_left_coords = (top_left_w, top_left_h)

    ground_truth_patch = original_image[top_left_h:top_left_h + patch_size, top_left_w:top_left_w + patch_size] 

    return original_image, ground_truth_patch, top_left_coords

def _plot_ground_truth_with_patch(original_image, top_left_coords, patch_size):
    ground_truth_fig, ax = plt.subplots()

    ax.imshow(original_image, cmap='gray')
    ax.axis("off")
    ax.set_title("Original Image")
    rect = matplotlib.patches.Rectangle(
        top_left_coords, patch_size, patch_size, 
        linewidth=2, edgecolor="orange", facecolor="none"
    )
    ax.add_patch(rect)
    ground_truth_fig.tight_layout()

    return ground_truth_fig

def _plot_patches(ground_truth_patches, noisy_patches, denoised_patches, titles=None):
    """
    Function to plot a ground truth patch, noisy patch, and multiple denoised patches.

    Args:
    - ground_truth_patch (ndarray): The ground truth image patch.
    - noisy_patch (ndarray): The noisy image patch.
    - denoised_patches (list of ndarray): List of denoised image patches.
    - titles (list of str, optional): List of titles for each plot. Defaults to None.
    """
    num_denoised = len(denoised_patches)

    fontsize = 16
    if num_denoised <= 2:
        # Create a figure with 1 row and (2 + num_denoised) columns
        fig, axes = plt.subplots(1, 2 + num_denoised, figsize=(5 * (num_denoised + 1), 5))

        # Plot ground truth patch
        axes[0].imshow(ground_truth_patches.squeeze(), cmap='gray')
        axes[0].axis("off")
        axes[0].set_title("Ground Truth Patch\n", fontdict={'fontsize': fontsize})

        # Plot noisy patch
        axes[1].imshow(noisy_patches.squeeze(), cmap='gray')
        axes[1].axis("off")
        axes[1].set_title("Noisy Patch\n", fontdict={'fontsize': fontsize})

        # Plot denoised patches with custom titles
        for i in range(num_denoised):
            axes[2 + i].imshow(denoised_patches[i].squeeze(), cmap='gray')
            axes[2 + i].axis("off")
            if titles:
                axes[2 + i].set_title(titles[i] + '\n', fontdict={'fontsize': fontsize})
            else:
                axes[2 + i].set_title(f"Denoised Patch {i + 1}\n", fontdict={'fontsize': fontsize})
    else:
        # Create a figure with (2 + num_denoised) rows and 2 columns (labels, images)
        fig, axes = plt.subplots(2 + num_denoised, 2, figsize=(10, 5 * (num_denoised + 1)))

        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i, j].axis("off")

        # First two rows are ground truth and noisy
        axes[0, 1].imshow(ground_truth_patches.squeeze(), cmap='gray')
        axes[1, 1].imshow(noisy_patches.squeeze(), cmap='gray')

        axes[0, 0].annotate("Ground Truth Patch", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=fontsize)
        axes[1, 0].annotate("Noisy Patch", xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center', fontsize=fontsize)
        
        # The remaining rows are denoised
        for i in range(num_denoised):
            if titles:
                axes[2 + i, 0].text(0.5, 0.5, titles[i], ha='center', va='center', fontsize=fontsize)
            else:
                axes[2 + i, 0].text(0.5, 0.5, "Denoised Patch {i + 1}", ha='center', va='center', fontsize=fontsize)

        for i in range(num_denoised):
            axes[2 + i, 1].imshow(denoised_patches[i].squeeze(), cmap='gray')

    fig.tight_layout()

    return fig

def _get_denoised_patch(noisy_patch, model):
    denoised_patch = model.predict(np.expand_dims(noisy_patch, axis=0))
    denoised_patch = np.clip(denoised_patch, 0, 1) * 255.0
    denoised_patch = denoised_patch.astype(np.uint8)

    return denoised_patch

def test_model_on_image(image_path, model, patch_coords=None, patch_size=192, 
                        zero_one_normalisation=True, greyscale=True,
                        stddev=None):
    # Get ground truth patch
    original_image, ground_truth_patch, top_left_coords = _image_to_ground_truth_patch(
        image_path, patch_coords, patch_size, zero_one_normalisation, greyscale
    )

    # Get noisy patch
    noisy_patch = noise_models.gaussian_noise(ground_truth_patch, stddev=stddev) 
    
    # Get denoised patch for the given model
    denoised_patch = _get_denoised_patch(noisy_patch, model)

    ground_truth_fig = _plot_ground_truth_with_patch(
        original_image, top_left_coords=top_left_coords, patch_size=patch_size
    )

    patches_fig = _plot_patches(ground_truth_patch, noisy_patch, denoised_patch)

    return ground_truth_fig, patches_fig

def compare_models_on_image(
        image_path, models, patch_coords=None,
        patch_size=192, zero_one_normalisation=True,
        greyscale=True, stddev=None, titles=None):

    # Get ground truth patch
    original_image, ground_truth_patch, top_left_coords = _image_to_ground_truth_patch(
        image_path, patch_coords, patch_size, zero_one_normalisation, greyscale
    )
    
    # Get noisy patch
    noisy_patch = noise_models.gaussian_noise(ground_truth_patch, stddev=stddev) 

    # Get denoised patches for each model
    denoised_patches = []
    for model in models:
        denoised_patches.append(_get_denoised_patch(noisy_patch, model))

    # Plot the original image with the patch rectangle
    ground_truth_fig = _plot_ground_truth_with_patch(
        original_image, top_left_coords=top_left_coords, patch_size=patch_size
    )

    patches_fig = _plot_patches(ground_truth_patch, noisy_patch, denoised_patches, titles) 

    return ground_truth_fig, patches_fig


def _plot_patches_multiple(ground_truth_patches, noisy_patches, denoised_patches, 
                           titles=None):
    # Number of images
    num_images = len(ground_truth_patches)
    
    # Determine the number of denoised patches per image (same for each image)
    num_denoised = len(denoised_patches[0])
    
    fontsize = 16
    
    # Create a figure with (num_denoised + 2) rows (labels + images)
    fig, axes = plt.subplots(num_denoised + 2, num_images + 1, figsize=(3.6 * (num_images), 4 * (num_denoised)))

    # Loop through the axes to remove axis ticks
    for i in range(num_denoised + 2):
        for j in range(num_images + 1):
            axes[i, j].axis("off")

    # Add labels in the first column
    axes[0, 0].text(0.5, 0.5, "Ground Truth Patch", ha='center', va='center', fontsize=fontsize, linespacing=2)
    axes[1, 0].text(0.5, 0.5, "Noisy Patch", ha='center', va='center', fontsize=fontsize, linespacing=2)
   
    for i in range(num_denoised):
        label = titles[i] if titles else f"Denoised Patch {i + 1}"
        axes[2 + i, 0].text(0.5, 0.5, label, ha='center', va='center', fontsize=fontsize, linespacing=2)

    # Plot the patches for each image
    for i in range(num_images):
        # Plot ground truth, noisy, and denoised patches for each image
        axes[0, i + 1].imshow(ground_truth_patches[i].squeeze(), cmap='gray')
        axes[1, i + 1].imshow(noisy_patches[i].squeeze(), cmap='gray')
        
        for j in range(num_denoised):
            axes[2 + j, i + 1].imshow(denoised_patches[i][j].squeeze(), cmap='gray')

    fig.tight_layout(pad=0)

    return fig

def compare_models_on_multiple_images(
        image_paths, models, patch_coords=None,
        patch_size=192, zero_one_normalisation=True,
        greyscale=True, stddev=None, titles=None):
    """
    Function to compare multiple models on multiple images.
    
    Args:
    - image_paths (list of str): List of image paths.
    - models (list of keras.Model): List of models to compare.
    - patch_coords (list of tuples): List of tuples representing the patches for each image.
    - patch_size (int): Size of the patch.
    - zero_one_normalisation (bool): Whether to normalize to [0, 1].
    - greyscale (bool): Whether to use greyscale images.
    - stddev (float, optional): Standard deviation for Gaussian noise.
    - titles (list of str, optional): Titles for each denoised patch.
    """
    ground_truth_patches = []
    noisy_patches = []
    denoised_patches = [[] for _ in range(len(image_paths))]

    # Process each image
    for i, image_path in enumerate(image_paths):
        # Get ground truth patch
        original_image, ground_truth_patch, top_left_coords = _image_to_ground_truth_patch(
            image_path, patch_coords[i], patch_size, zero_one_normalisation, greyscale
        )
        
        # Get noisy patch
        noisy_patch = noise_models.gaussian_noise(ground_truth_patch, stddev=stddev) 

        # Get denoised patches for each model
        for model in models:
            denoised_patch = _get_denoised_patch(noisy_patch, model)
            denoised_patches[i].append(denoised_patch)

        # Add the ground truth and noisy patches to the respective lists
        ground_truth_patches.append(ground_truth_patch)
        noisy_patches.append(noisy_patch)

    # Plot the images with their patches
    patches_fig = _plot_patches_multiple(
        ground_truth_patches, 
        noisy_patches, 
        denoised_patches, 
        titles
    )

    return patches_fig
