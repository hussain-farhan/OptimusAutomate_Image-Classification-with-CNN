"""
app.py
Streamlit frontend for visualizing CIFAR-10 predictions and evaluation results.

Run from the project root:
    streamlit run src/app.py
"""

import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

from train import CLASS_NAMES, MODEL_PATH, OUTPUTS_DIR, load_data

ACC_LOSS_PLOT_PATH = os.path.join(OUTPUTS_DIR, "accuracy_loss_plot.png")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
SAMPLE_PRED_PATH = os.path.join(OUTPUTS_DIR, "sample_predictions.png")

st.set_page_config(
    page_title="CIFAR-10 CNN Classifier",
    page_icon="🖼️",
    layout="wide",
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalize an uploaded image for the CNN (32×32, RGB, [0, 1])."""
    img = image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict_image(model, image: Image.Image):
    batch = preprocess_image(image)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def render_prediction_card(image, true_label, pred_idx, probs, key_prefix=""):
    col_img, col_info = st.columns([1, 1])

    with col_img:
        st.image(image, use_container_width=True)

    with col_info:
        pred_label = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx] * 100

        if true_label is not None:
            correct = true_label == pred_label
            st.markdown(f"**True label:** {true_label}")
            st.markdown(
                f"**Prediction:** :{'green' if correct else 'red'}[{pred_label}] "
                f"({confidence:.1f}%)"
            )
        else:
            st.markdown(f"**Prediction:** **{pred_label}** ({confidence:.1f}%)")

        chart_data = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(chart_data)


def page_upload(model):
    st.header("Upload an Image")
    st.caption("Upload any image — it will be resized to 32×32 for the CIFAR-10 model.")

    uploaded = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
    )

    if uploaded is None:
        st.info("Upload an image to see the model's prediction.")
        return

    image = Image.open(uploaded)
    pred_idx, probs = predict_image(model, image)
    render_prediction_card(image, None, pred_idx, probs)


def page_gallery(model):
    st.header("CIFAR-10 Test Gallery")
    st.caption("Browse random images from the CIFAR-10 test set with live predictions.")

    count = st.slider("Number of images", min_value=4, max_value=20, value=8, step=4)
    if st.button("Show new random images", type="primary"):
        st.session_state["gallery_seed"] = np.random.randint(0, 1_000_000)

    seed = st.session_state.get("gallery_seed", 42)
    rng = np.random.default_rng(seed)

    (_, _), (x_test, y_test) = load_data()
    indices = rng.choice(len(x_test), size=count, replace=False)

    cols = st.columns(4)
    for i, idx in enumerate(indices):
        img_arr = (x_test[idx] * 255).astype(np.uint8)
        image = Image.fromarray(img_arr)
        true_label = CLASS_NAMES[int(y_test[idx])]

        batch = np.expand_dims(x_test[idx], axis=0)
        probs = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        with cols[i % 4]:
            st.image(image, caption=f"True: {true_label}", use_container_width=True)
            pred_label = CLASS_NAMES[pred_idx]
            conf = probs[pred_idx] * 100
            color = "green" if pred_label == true_label else "red"
            st.markdown(
                f":{color}[**{pred_label}**] ({conf:.0f}%)"
            )


def page_results():
    st.header("Training & Evaluation Results")

    plots = [
        ("Accuracy & Loss", ACC_LOSS_PLOT_PATH),
        ("Confusion Matrix", CONFUSION_MATRIX_PATH),
        ("Sample Predictions", SAMPLE_PRED_PATH),
    ]

    available = [(title, path) for title, path in plots if os.path.exists(path)]

    if not available:
        st.warning(
            "No evaluation plots found. Run `python src/evaluate.py` after training."
        )
        return

    for title, path in available:
        st.subheader(title)
        st.image(path, use_container_width=True)


def main():
    st.title("CIFAR-10 Image Classifier")
    st.markdown(
        "Visual interface for the CNN that classifies images into "
        "**airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck**."
    )

    model = load_model()
    if model is None:
        st.error(
            f"No trained model found at `{MODEL_PATH}`. "
            "Run `python src/train.py` first."
        )
        st.stop()

    tab_upload, tab_gallery, tab_results = st.tabs(
        ["Upload & Classify", "CIFAR-10 Gallery", "Training Results"]
    )

    with tab_upload:
        page_upload(model)

    with tab_gallery:
        page_gallery(model)

    with tab_results:
        page_results()


if __name__ == "__main__":
    main()
