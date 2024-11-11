import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def check_gpu_use():
    if len(tf.config.list_physical_devices('GPU')) == 0:
        raise Exception('****** NO GPU DETECTED! ******')
    print('GPU DETECTED \u2713')
    print(tf.config.list_physical_devices('GPU'))


def calculate_metrics(evals: list):
    mean = np.mean(evals, axis=0)
    return mean[0], mean[1] # loss, acc

def get_loss_curve(history: keras.callbacks.History):
    loss = history.history['loss']
    epochs = np.arange(1, len(loss) + 1, dtype=int)

    fig, ax = plt.subplots()

    ax.plot(epochs, loss, 'bo-')
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_title('Loss over epochs')

    return fig

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

    if overlay:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(14, 5))

        # Plot each fold
        for fold_i in range(N_FOLDS):
            fold_epochs = np.arange(1, len(histories[fold_i].history['acc']) + 1, dtype=int)
            ax[0].plot(fold_epochs, accuracies[fold_i][:len(fold_epochs)], 'o--b', alpha=0.3)
            ax[0].plot(fold_epochs, val_accuracies[fold_i][:len(fold_epochs)], 'o--', color='orange', alpha=0.3)
            ax[1].plot(fold_epochs, losses[fold_i][:len(fold_epochs)], 'o--b', alpha=0.3)
            ax[1].plot(fold_epochs, val_losses[fold_i][:len(fold_epochs)], 'o--', color='orange', alpha=0.3)
        
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

    if overlay:
        if max_epochs == 1:
            print("WARNING: plot is identical regardless of overlay setting for epochs=1")

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(14, 5))

        # Plot each epoch
        for epoch_i in range(max_epochs):
            ax[0].plot(folds, epoch_accuracies[epoch_i], 'o--b', alpha=0.3, label=f'Epoch {epoch_i + 1}')
            ax[0].plot(folds, epoch_val_accuracies[epoch_i], 'o--', color='orange', alpha=0.3)

            ax[1].plot(folds, epoch_losses[epoch_i], 'o--b', alpha=0.3)
            ax[1].plot(folds, epoch_val_losses[epoch_i], 'o--', color='orange', alpha=0.3)

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