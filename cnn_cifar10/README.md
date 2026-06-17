# CNN Image Classification — CIFAR-10

A Convolutional Neural Network that classifies images into 10 categories using the CIFAR-10 dataset, with data augmentation, dropout regularization, and full evaluation (accuracy/loss plots + confusion matrix).

## Project structure

```
cnn-cifar10/
│
├── data/                       # Cached dataset (auto-generated, gitignored)
│
├── outputs/                    # Trained model + generated plots
│   ├── accuracy_loss_plot.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
│
├── src/
│   ├── model.py                # CNN architecture definition
│   ├── train.py                 # Data loading, augmentation, training loop
│   └── evaluate.py              # Evaluation, plots, confusion matrix
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Overview

This project trains a CNN from scratch to classify 32×32 color images into 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

**Architecture:** 3 convolutional blocks (32 → 64 → 128 filters) with batch normalization, max pooling, and dropout, followed by a dense classifier head with softmax output.

**Expected performance:** ~78–82% test accuracy after 30 epochs.

## Requirements

- Python 3.8+
- ~200 MB free disk space (for the dataset cache)
- Internet connection (first run only, to download CIFAR-10)

## Installation

```bash
git clone <your-repo-url>
cd cnn-cifar10
pip install -r requirements.txt
```

## Usage

Run the two stages in order, from the project root.

**1. Train the model**

```bash
python src/train.py
```

This will:
- Download CIFAR-10 automatically on first run, and cache it locally in `data/cifar10.npz`
- Normalize the data and apply data augmentation (flips, rotations, shifts, zoom) during training
- Build and train the CNN for 30 epochs
- Save the best model weights to `outputs/best_cnn_cifar10.keras`
- Save training history to `outputs/history.pkl` (used by evaluate.py)

**2. Evaluate the model**

```bash
python src/evaluate.py
```

This will:
- Load the trained model and the saved training history
- Print test accuracy, loss, and a full classification report
- Generate `outputs/accuracy_loss_plot.png`
- Generate `outputs/confusion_matrix.png`
- Generate `outputs/sample_predictions.png`

## Module breakdown

| File | Responsibility |
|---|---|
| `src/model.py` | Defines `build_cnn()` and `compile_model()` — pure architecture, no data or training logic |
| `src/train.py` | Loads/caches CIFAR-10, sets up data augmentation, trains the model, saves weights + history |
| `src/evaluate.py` | Loads the trained model, runs predictions on the test set, generates all plots and reports |

You can also import and reuse pieces independently, for example:

```python
from model import build_cnn, compile_model

model = build_cnn()
compile_model(model)
model.summary()
```

## Dataset

CIFAR-10 is loaded automatically — no manual download required. On first run, `train.py` downloads it via Keras and caches a local copy at `data/cifar10.npz` so future runs don't re-download.

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

Key hyperparameters you can adjust:

| Parameter | Default | Location |
|---|---|---|
| Epochs | 30 | `EPOCHS` in `src/train.py` |
| Batch size | 64 | `BATCH_SIZE` in `src/train.py` |
| Learning rate | 0.001 | `compile_model()` call in `src/train.py` |
| Dropout (conv blocks) | 0.25 | `src/model.py` |
| Dropout (dense head) | 0.5 | `src/model.py` |

## Notes

- Training runs on CPU but is significantly faster on GPU. TensorFlow will use a GPU automatically if available (CUDA + cuDNN installed).
- The best model (by validation accuracy) is checkpointed during training, so evaluation always uses the best-performing weights, not necessarily the last epoch's.
- `data/` and the contents of `outputs/` are gitignored by default since they're large generated artifacts — adjust `.gitignore` if you want to commit them.
- To improve accuracy beyond ~82%, consider increasing epochs to 50–100 or switching to a pretrained backbone (e.g., `keras.applications.ResNet50`) with fine-tuning.

## License

MIT — see `LICENSE`.
