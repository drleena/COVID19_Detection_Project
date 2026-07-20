import os
import gdown
import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="COVID-19 Detection",
    page_icon="🩺",
    layout="centered"
)

MODEL_FILE = "best_resnet50_model.keras"

FILE_ID = "1FMb-74zY4j9coUiGtMuRnGAE84Tbxj0f"

URL = f"https://drive.google.com/uc?id={FILE_ID}"

try:
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Downloading trained model... Please wait..."):
            gdown.download(URL, MODEL_FILE, quiet=False)

except Exception as e:
    st.error(f"Unable to download the model.\n\n{e}")
    st.stop()


@st.cache_resource
def load_covid_model():
    return load_model(MODEL_FILE)


model = load_covid_model()

classes = [
    "COVID",
    "Normal",
    "Viral Pneumonia"
]

st.title("🩺 COVID-19 Chest X-ray Classification")

st.write(
    """
Upload a chest X-ray image and the model will predict whether it belongs to:

- COVID-19
- Normal
- Viral Pneumonia
"""
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
    )

    img = image.resize((224, 224))

    img = np.array(img).astype("float32")

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    with st.spinner("Analyzing X-ray..."):
        prediction = model.predict(img, verbose=0)

    pred_class = np.argmax(prediction)

    confidence = np.max(prediction)

    st.success(
        f"Prediction: {classes[pred_class]}"
    )

    st.info(
        f"Confidence: {confidence * 100:.2f}%"
    )

    st.subheader("Prediction Probabilities")

    for i, c in enumerate(classes):

        prob = float(prediction[0][i])

        st.write(f"**{c}**")

        st.progress(prob)

        st.write(f"{prob * 100:.2f}%")
