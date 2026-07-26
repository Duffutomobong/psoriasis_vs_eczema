"""
app.py
------
GET 324 - Laboratory Exercise 10 (Mini-Project)
Streamlit web application for deploying the Psoriasis vs Eczema
binary image classifier.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub (include model/psoriasis_eczema_model.keras
       and model/class_indices.json, or use Git LFS if the model is large).
    2. Go to https://share.streamlit.io, connect your GitHub repo,
       set the main file path to app.py, and deploy.
"""

import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "model/psoriasis_eczema_model.keras"
CLASS_INDICES_PATH = "model/class_indices.json"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Psoriasis vs Eczema Classifier",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    st.title("🩺 Psoriasis vs Eczema Classifier")
    st.write(
        "Upload a close-up image of a skin lesion and the model will predict "
        "whether it is more consistent with **Psoriasis** or **Eczema**."
    )

    st.info(
        "⚠️ **Disclaimer**: This tool is an academic mini-project for course "
        "GET 324. It is **not** a medical device and must not be used for "
        "real diagnosis. Always consult a qualified dermatologist.",
        icon="⚠️",
    )

    try:
        model, idx_to_class = load_model_and_classes()
    except Exception as e:
        st.error(
            "Could not load the trained model. Make sure "
            f"`{MODEL_PATH}` and `{CLASS_INDICES_PATH}` exist "
            "(run train.py first). \n\nDetails: " + str(e)
        )
        return

    uploaded_file = st.file_uploader(
        "Choose a skin lesion image...", type=["jpg", "jpeg", "png"]
    )

    col1, col2 = st.columns(2)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            input_arr = preprocess_image(image)
            prob = float(model.predict(input_arr)[0][0])

            # idx_to_class maps 0/1 -> class name (alphabetical: 0=eczema, 1=psoriasis)
            positive_class = idx_to_class[1]
            negative_class = idx_to_class[0]

            if prob >= 0.5:
                predicted_class = positive_class
                confidence = prob
            else:
                predicted_class = negative_class
                confidence = 1 - prob

        with col2:
            st.subheader("Prediction")
            st.markdown(f"### **{predicted_class.upper()}**")
            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.progress(confidence)

            st.write("---")
            st.write("Class probabilities:")
            st.write(f"- {negative_class.capitalize()}: {(1 - prob) * 100:.1f}%")
            st.write(f"- {positive_class.capitalize()}: {prob * 100:.1f}%")
    else:
        st.write("👆 Upload an image to get a prediction.")

    st.write("---")
    with st.expander("About this project"):
        st.write(
            "This application was developed for GET 324 Laboratory Exercise "
            "10 (Mini-Project). It uses a MobileNetV2 transfer-learning "
            "convolutional neural network fine-tuned on labeled Psoriasis "
            "and Eczema skin lesion images to perform binary classification."
        )


if __name__ == "__main__":
    main()
