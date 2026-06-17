"""
evaluate.py
Loads the trained model + saved history, and produces:
    - accuracy_loss_plot.png
    - confusion_matrix.png
    - sample_predictions.png
    - a printed classification report

Run from the project root, after train.py has completed:
    python src/evaluate.py
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras

from train import load_data, MODEL_PATH, HISTORY_PATH, OUTPUTS_DIR, CLASS_NAMES

ACC_LOSS_PLOT_PATH   = os.path.join(OUTPUTS_DIR, "accuracy_loss_plot.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
SAMPLE_PRED_PATH      = os.path.join(OUTPUTS_DIR, "sample_predictions.png")


def plot_accuracy_loss(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Training History — CIFAR-10", fontsize=14, fontweight="bold")

    axes[0].plot(history["accuracy"], label="Train accuracy", color="#2196F3")
    axes[0].plot(history["val_accuracy"], label="Validation accuracy",
                 color="#FF5722", linestyle="--")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["loss"], label="Train loss", color="#4CAF50")
    axes[1].plot(history["val_loss"], label="Validation loss",
                 color="#9C27B0", linestyle="--")
    axes[1].set_title("Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ACC_LOSS_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {ACC_LOSS_PLOT_PATH}")


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Confusion Matrix — CIFAR-10 CNN", fontsize=14, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Raw counts")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(cm_normalized, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5)
    axes[1].set_title("Normalized (% per class)")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {CONFUSION_MATRIX_PATH}")


def plot_sample_predictions(x_test, y_test, y_pred, y_pred_probs):
    fig, axes = plt.subplots(3, 5, figsize=(14, 9))
    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")

    indices = np.random.choice(len(x_test), 15, replace=False)
    for i, idx in enumerate(indices):
        ax = axes[i // 5][i % 5]
        ax.imshow(x_test[idx])

        true_lbl = CLASS_NAMES[y_test[idx]]
        pred_lbl = CLASS_NAMES[y_pred[idx]]
        conf = y_pred_probs[idx][y_pred[idx]] * 100

        color = "green" if true_lbl == pred_lbl else "red"
        ax.set_title(f"True: {true_lbl}\nPred: {pred_lbl} ({conf:.0f}%)",
                     color=color, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(SAMPLE_PRED_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {SAMPLE_PRED_PATH}")


def evaluate():
    # Load test data
    (_, _), (x_test, y_test) = load_data()

    # Load trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run src/train.py first."
        )
    model = keras.models.load_model(MODEL_PATH)

    # Load training history
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(
            f"No training history found at {HISTORY_PATH}. Run src/train.py first."
        )
    with open(HISTORY_PATH, "rb") as f:
        history = pickle.load(f)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}\n")

    # Predictions
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Plots
    plot_accuracy_loss(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_sample_predictions(x_test, y_test, y_pred, y_pred_probs)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))


if __name__ == "__main__":
    evaluate()
