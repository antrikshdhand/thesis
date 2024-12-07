import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches
import cv2

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

def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

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

def get_psnr_and_loss_curves(history: keras.callbacks.History, together=False):
    # Extract epochs
    num_epochs = len(history.history['psnr'])
    epochs = np.arange(1, num_epochs + 1, dtype=int)

    if together:
        # Create a single figure with 2 subplots side by side
        fig, (ax_psnr, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))
        
        # PSNR Plot
        ax_psnr.plot(epochs, history.history['psnr'], label='PSNR', marker='o')
        ax_psnr.plot(epochs, history.history['val_psnr'], label='Validation PSNR', marker='s')
        ax_psnr.set_title('PSNR over epochs')
        ax_psnr.set_xlabel('Epochs')
        ax_psnr.set_ylabel('PSNR (dB)')
        ax_psnr.legend()
        ax_psnr.grid(True)

        # Loss Plot
        ax_loss.plot(epochs, history.history['loss'], label='Loss', marker='o')
        ax_loss.plot(epochs, history.history['val_loss'], label='Validation loss', marker='s')
        ax_loss.set_title('Loss over epochs')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        fig.tight_layout()

        return fig
    
    else:
        # Separate figures for PSNR and Loss
        fig_psnr, ax_psnr = plt.subplots()
        ax_psnr.plot(epochs, history.history['psnr'], label='PSNR', marker='o')
        ax_psnr.plot(epochs, history.history['val_psnr'], label='Validation PSNR', marker='s')
        ax_psnr.set_title('PSNR over epochs')
        ax_psnr.set_xlabel('Epochs')
        ax_psnr.set_ylabel('PSNR (dB)')
        ax_psnr.legend()
        ax_psnr.grid(True)

        # Loss Plot
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(epochs, history.history['loss'], label='Loss', marker='o')
        ax_loss.plot(epochs, history.history['val_loss'], label='Validation loss', marker='s')
        ax_loss.set_title('Loss over epochs')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True)

        return fig_psnr, fig_loss 

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


def image_to_ground_truth_patch(image_path, patch_coords=None, patch_size=192,
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


def plot_ground_truth_with_patch(original_image, top_left_coords, patch_size):
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

def plot_patches(ground_truth_patch, noisy_patch, denoised_patch):
    patches_fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(ground_truth_patch.squeeze(), cmap='gray')
    axes[0].axis("off")
    axes[0].set_title("Ground Truth Patch\n", fontdict={'fontsize':16})

    axes[1].imshow(noisy_patch.squeeze(), cmap='gray')
    axes[1].axis("off")
    axes[1].set_title("Noisy Patch\n", fontdict={'fontsize':16})

    axes[2].imshow(denoised_patch.squeeze(), cmap='gray')
    axes[2].axis("off")
    axes[2].set_title("Denoised Patch\n", fontdict={'fontsize':16})

    patches_fig.tight_layout()

    return patches_fig

def test_model_on_image(image_path, model, patch_coords=None, patch_size=192, 
                        zero_one_normalisation=True, greyscale=True,
                        stddev=None):
    original_image, ground_truth_patch, top_left_coords = image_to_ground_truth_patch(
        image_path, patch_coords, patch_size, zero_one_normalisation, greyscale
    )
    noisy_patch = noise_models.gaussian_noise(ground_truth_patch, stddev=stddev) 
    
    denoised_patch = model.predict(np.expand_dims(noisy_patch, axis=0))
    denoised_patch = np.clip(denoised_patch, 0, 1) * 255.0
    denoised_patch = denoised_patch.astype(np.uint8)

    ground_truth_fig = plot_ground_truth_with_patch(
        original_image, top_left_coords=top_left_coords, patch_size=patch_size
    )

    patches_fig = plot_patches(ground_truth_patch, noisy_patch, denoised_patch)

    return ground_truth_fig, patches_fig


def compare_two_models_denoising(
        image_path, model1, model2, patch_coords=None,
        patch_size=192, zero_one_normalisation=True,
        greyscale=True, stddev=None, model1_name=None, model2_name=None):

    original_image, ground_truth_patch, top_left_coords = image_to_ground_truth_patch(
        image_path, patch_coords, patch_size, zero_one_normalisation, greyscale
    )
    
    noisy_patch = noise_models.gaussian_noise(ground_truth_patch, stddev=stddev) 

    denoised_patch_1 = model1.predict(np.expand_dims(noisy_patch, axis=0))
    denoised_patch_1 = np.clip(denoised_patch_1, 0, 1) * 255.0
    denoised_patch_1 = denoised_patch_1.astype(np.uint8)
    
    denoised_patch_2 = model2.predict(np.expand_dims(noisy_patch, axis=0))
    denoised_patch_2 = np.clip(denoised_patch_2, 0, 1) * 255.0
    denoised_patch_2 = denoised_patch_2.astype(np.uint8)

    # Plot the original image with the patch rectangle
    ground_truth_fig = plot_ground_truth_with_patch(
        original_image, top_left_coords=top_left_coords, patch_size=patch_size
    )

    # Plot the ground truth patch, noisy patch, and both denoised patches
    patches_fig, axes = plt.subplots(1, 4, figsize=(14, 5))

    fontsize = 17

    axes[0].imshow(ground_truth_patch.squeeze(), cmap='gray')
    axes[0].axis("off")
    axes[0].set_title("Ground Truth Patch\n", fontdict={'fontsize':fontsize})

    axes[1].imshow(noisy_patch.squeeze(), cmap='gray')
    axes[1].axis("off")
    axes[1].set_title("Noisy Patch\n", fontdict={'fontsize':fontsize})

    # Plot Denoised Patch from Model 1
    axes[2].imshow(denoised_patch_1.squeeze(), cmap='gray')
    axes[2].axis("off")
    if model1_name is None:
        axes[2].set_title("Denoised Patch (Model 1)\n", fontdict={'fontsize':fontsize})
    else:
        axes[2].set_title(f"Denoised Patch ({model1_name})\n", fontdict={'fontsize':fontsize})

    # Plot Denoised Patch from Model 2
    axes[3].imshow(denoised_patch_2.squeeze(), cmap='gray')
    axes[3].axis("off")
    if model2_name is None:
        axes[3].set_title("Denoised Patch (Model 2)\n", fontdict={'fontsize':fontsize})
    else:
        axes[3].set_title(f"Denoised Patch ({model2_name})\n", fontdict={'fontsize':fontsize})

    patches_fig.tight_layout(pad=0.2)
    # patches_fig.subplots_adjust(top=0.85)

    return ground_truth_fig, patches_fig
