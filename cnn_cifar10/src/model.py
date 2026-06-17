"""
model.py
Defines the CNN architecture for CIFAR-10 classification.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Builds a CNN with 3 convolutional blocks and a dense classifier head.

    Architecture:
        Block 1: Conv(32) -> BN -> Conv(32) -> BN -> Pool -> Dropout(0.25)
        Block 2: Conv(64) -> BN -> Conv(64) -> BN -> Pool -> Dropout(0.25)
        Block 3: Conv(128)-> BN -> Conv(128)-> BN -> Pool -> Dropout(0.25)
        Head:    Flatten -> Dense(256) -> BN -> Dropout(0.5) -> Dense(num_classes, softmax)

    Args:
        input_shape: shape of input images, default (32, 32, 3) for CIFAR-10
        num_classes: number of output classes, default 10

    Returns:
        An uncompiled keras.Sequential model.
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compiles the model with Adam optimizer and sparse categorical crossentropy.

    Args:
        model: an uncompiled keras model
        learning_rate: optimizer learning rate, default 0.001

    Returns:
        The compiled model (compiled in place, also returned for convenience).
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    # Quick sanity check: build and print the model summary
    model = build_cnn()
    compile_model(model)
    model.summary()
