"""
train.py
Loads CIFAR-10, applies data augmentation, builds the CNN (from model.py),
trains it, and saves the best weights + training history.

Run from the project root:
    python src/train.py
"""

import os
import pickle

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_cnn, compile_model

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(OUTPUTS_DIR, "best_cnn_cifar10.keras")
HISTORY_PATH = os.path.join(OUTPUTS_DIR, "history.pkl")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

EPOCHS     = 30
BATCH_SIZE = 64


def load_data():
    """
    Loads CIFAR-10 via Keras (auto-downloads to ~/.keras/datasets on first run,
    cached afterward). Also caches a local copy of the raw arrays under data/
    so subsequent loads don't depend on the Keras cache location.
    """
    cache_file = os.path.join(DATA_DIR, "cifar10.npz")

    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        npz = np.load(cache_file)
        x_train, y_train = npz["x_train"], npz["y_train"]
        x_test, y_test   = npz["x_test"], npz["y_test"]
    else:
        print("Downloading CIFAR-10 via Keras (first run only)...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test  = y_test.flatten()

        print(f"Caching dataset locally to {cache_file}")
        np.savez_compressed(
            cache_file,
            x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test
        )

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    return (x_train, y_train), (x_test, y_test)


def build_augmenter():
    """Returns an ImageDataGenerator for training-time data augmentation."""
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest"
    )
    return datagen


def train():
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    datagen = build_augmenter()
    datagen.fit(x_train)

    model = build_cnn()
    compile_model(model)
    model.summary()

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
    )

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[lr_scheduler, checkpoint],
        verbose=1
    )

    # Save training history so evaluate.py can plot it without retraining
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)

    print(f"\nBest model saved to: {MODEL_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")


if __name__ == "__main__":
    train()
