# CNN Image Classification — CIFAR-10

A Convolutional Neural Network that classifies images into 10 categories using the CIFAR-10 dataset, with data augmentation, dropout regularization, and full evaluation (accuracy/loss plots + confusion matrix).

## Overview

This project trains a CNN from scratch to classify 32×32 color images into 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

**Architecture:** 3 convolutional blocks (32 → 64 → 128 filters) with batch normalization, max pooling, and dropout, followed by a dense classifier head with softmax output.

**Expected performance:** ~78-82% test accuracy after 30 epochs.

## Files

| File | Description |
|---|---|
| `cnn_cifar10.py` | Main script — the only file you need to run |
| `best_cnn_cifar10.keras` | Saved model weights (generated after training) |
| `accuracy_loss_plot.png` | Training/validation accuracy and loss curves (generated) |
| `confusion_matrix.png` | Raw and normalized confusion matrices (generated) |
| `sample_predictions.png` | Grid of sample predictions with true/predicted labels (generated) |

## Requirements

- Python 3.8+
- ~200 MB free disk space (for the dataset cache)
- Internet connection (first run only, to download CIFAR-10)

## Installation

```bash
pip install tensorflow scikit-learn seaborn matplotlib numpy
```

## Usage

```bash
python cnn_cifar10.py
```

That's it. The script handles everything:

1. Downloads CIFAR-10 automatically on first run (cached afterward at `~/.keras/datasets/`)
2. Normalizes and preprocesses the data
3. Applies data augmentation (flips, rotations, shifts, zoom) during training
4. Builds and trains the CNN for 30 epochs
5. Saves the best model weights based on validation accuracy
6. Generates accuracy/loss plots, a confusion matrix, a classification report, and sample predictions

## Dataset

CIFAR-10 is loaded directly through Keras — no manual download required:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
```

| Split | Images | Shape |
|---|---|---|
| Train | 50,000 | (50000, 32, 32, 3) |
| Test | 10,000 | (10000, 32, 32, 3) |

## Model architecture

```
Input (32×32×3)
  → Conv Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  → Conv Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  → Conv Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
  → Flatten → Dense(256) → BN → Dropout(0.5)
  → Dense(10, softmax)
```

## Configuration

Key hyperparameters you can adjust at the top of `cnn_cifar10.py`:

| Parameter | Default | Location |
|---|---|---|
| Epochs | 30 | `EPOCHS` variable |
| Batch size | 64 | `BATCH_SIZE` variable |
| Learning rate | 0.001 | `Adam(learning_rate=...)` |
| Dropout (conv blocks) | 0.25 | `Dropout(0.25)` |
| Dropout (dense head) | 0.5 | `Dropout(0.5)` |

## Output

After running, you'll see in the console:

- Model summary (layer-by-layer parameter counts)
- Training progress per epoch (loss/accuracy for train and validation)
- Final test accuracy and loss
- A full classification report (precision, recall, F1-score per class)

And in the project folder:

- `accuracy_loss_plot.png` — training curves
- `confusion_matrix.png` — which classes get confused with which
- `sample_predictions.png` — visual sanity check of 15 random predictions

## Notes

- Training runs on CPU but is significantly faster on GPU. TensorFlow will use GPU automatically if available (CUDA + cuDNN installed).
- The best model (by validation accuracy) is checkpointed during training, so the final evaluation always uses the best-performing weights, not necessarily the last epoch's.
- To improve accuracy beyond ~82%, consider increasing epochs to 50-100 or switching to a pretrained backbone (e.g., `keras.applications.ResNet50`) with fine-tuning.
