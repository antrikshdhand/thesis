import os
import time

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {DEVICE}")


def check_step(idx: int, percent_steps: np.array):
    return (np.where(percent_steps == idx)[0][0] + 1) * 10


def format_time(duration: float):
    minutes, seconds = divmod(duration, 60)
    return f"{int(minutes)}min {seconds:.1f}sec"


train_acc, test_acc = [], []
train_loss, test_loss = [], []


def evaluate(network, criterion, test_ds, batch_size):
    correct = 0.0
    incorrect = 0.0
    running_loss = 0.0

    for i, data in enumerate(test_ds, 0):
        features, labels, filenames = data
        features = features.to(DEVICE)
        features = features.unsqueeze(1)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            outputs = network(features)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            c = torch.argmax(outputs, dim=1).eq(labels).float()
            correct += c.sum().item()
            incorrect += torch.argmax(outputs, dim=1).not_equal(labels).sum().item()

    ns = len(test_ds) * batch_size
    r_loss = running_loss / ns
    accuracy = (correct / ns) * 100

    test_acc.append(accuracy)
    test_loss.append(r_loss)

    print(f"[Validation] [accuracy: {accuracy:.3f}] [loss: {r_loss:.3f}]")

    return accuracy, r_loss


def train_epoch(network, criterion, optimizer, epoch, train_ds, batch_size, val_ds=None):
    print(f"======= Starting Epoch {epoch + 1} =======")

    percent_steps = np.rint(np.arange(10, 110, 10) * 0.01 * len(train_ds)).astype(int)
    percent_steps[-1] -= 1

    start_time = time.perf_counter()

    running_loss = 0.0
    correct = 0.0
    accuracy = 0.0
    r_loss = 0.0

    for idx, data in enumerate(train_ds, 0):
        features, labels, filenames = data
        features = features.to(DEVICE)
        features = features.unsqueeze(1)
        labels = labels.to(DEVICE)

        outputs = network(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += torch.argmax(outputs, dim=1).eq(labels).sum().item()
        running_loss += loss.item()

        if idx in percent_steps:
            pc = check_step(idx, percent_steps)
            elapsed = time.perf_counter() - start_time
            ns = idx * batch_size
            r_loss = running_loss / ns
            accuracy = (correct / ns) * 100
            print(
                f"[{epoch + 1}, {idx + 1:5d}] [{pc}% complete] [duration: {format_time(elapsed)}] [accuracy: {accuracy:.3f}] [loss: {r_loss:.3f}]")

    return accuracy, r_loss


def training_validation_plot(training_acc, training_loss, validation_acc, validation_loss, outpath):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, layout='constrained')

    x_ticks = np.arange(0, len(training_acc), 1)
    x_labels = x_ticks + 1  # epochs

    accuracy_minor_labels = np.arange(0, 102.5, 2.5)

    accuracy_major_labels = np.arange(0, 110.0, 10.0)

    fig.axes[0].set_title('Train Test Accuracy & Loss')
    fig.axes[0].plot(training_acc, '.-b', label='Training')
    fig.axes[0].plot(validation_acc, color='darkorange', marker='.', label='Validation')

    fig.axes[0].grid(True, which='major')
    fig.axes[0].set_xticks(x_ticks, labels=x_labels)
    fig.axes[0].set_yticks(accuracy_major_labels)
    fig.axes[0].set_yticks(accuracy_minor_labels, minor=True)
    fig.axes[0].set_ylabel('Accuracy (%)')
    fig.axes[0].legend()

    # fig.axes[1].set_title('Loss')
    fig.axes[1].plot(training_loss, color='b', marker='.', label='Training', linestyle='--')
    fig.axes[1].plot(validation_loss, color='darkorange', marker='.', label='Validation', linestyle='--')
    fig.axes[1].grid(True)
    fig.axes[1].grid(True, which='minor')
    fig.axes[1].set_xlabel('Epoch')
    y_min = 0.0
    y_max = 0.20
    step = 0.05
    loss_major = np.arange(y_min, y_max + step, step)
    fig.axes[1].set_yticks(loss_major)
    # fig.axes[1].yaxis.set_major_locator(MultipleLocator(y_max/10))
    fig.axes[1].yaxis.set_minor_locator(MultipleLocator(step / 4))

    fig.axes[1].set_ylabel('Cross-Entropy Loss')
    fig.axes[1].legend()

    fig.savefig(os.path.join(outpath, "accuracy_loss_plots.png"))

