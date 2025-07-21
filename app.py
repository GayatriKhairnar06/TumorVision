import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = './mobilenet_v2_best.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at: {MODEL_PATH}. Please ensure it's in the correct directory.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_tumor_type(model, img_array, class_names):
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    return predicted_class_name, confidence, predictions[0]

# --- Streamlit UI ---
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Image Classifier")
st.markdown("Upload a brain MRI image to get an AI-powered tumor classification.")

# --- Visible Tumor Info ---
st.markdown("""
### 🔍 Tumor Type Overview:
- **Glioma**: Tumors originating from glial cells in the brain or spine.
- **Meningioma**: Tumors developing from the meninges (the membranes that surround the brain and spinal cord).
- **Pituitary Tumor**: Abnormal growths in the pituitary gland, affecting hormonal balance.
- **No Tumor**: No tumor detected in the scan.

> 🔬 **Disclaimer**: This tool is for educational purposes only and is not intended for medical diagnosis.
""")

model = load_model()

if model:
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded MRI Image', use_container_width=True)
        st.write("Classifying...")

        try:
            temp_file_path = "temp_image.png"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            img_array = preprocess_image(temp_file_path)
            predicted_class, confidence, all_predictions = pr
