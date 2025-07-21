import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import base64

# Title and background
def set_background(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background("static/background.png")  # Make sure this path is correct

st.title("🧠 Brain Tumor Classification")
st.markdown("Upload an MRI image to detect the type of brain tumor.")

CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Load your model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Dropdown always visible
st.sidebar.title("About")
st.sidebar.markdown(
    """
    - 👩‍⚕️ **Model**: Deep Learning (CNN)
    - 🧪 **Classes**:
        - Glioma
        - Meningioma
        - No Tumor
        - Pituitary
    - 📩 Contact: khairnargayatri333@gmail.com
    """
)

# Uploading image
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "jpeg", "png"])

# Preprocess function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Prediction function
def predict_tumor_type(model, img_array, class_names):
    predictions = model.predict(img_array)
    confidence = 100 * np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class, confidence, predictions[0]

# Prediction block
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    try:
        temp_file_path = "temp_image.png"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img_array = preprocess_image(temp_file_path)
        predicted_class, confidence, all_predictions = predict_tumor_type(model, img_array, CLASS_NAMES)

        st.success(f"Prediction: **{predicted_class.replace('_', ' ').title()}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        st.subheader("All Class Probabilities:")
        for class_name, prob in zip(CLASS_NAMES, all_predictions):
            st.write(f"- {class_name.replace('_', ' ').title()}: **{prob * 100:.2f}%**")

        os.remove(temp_file_path)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.warning("Please ensure the uploaded file is a valid image (JPG, JPEG, PNG).")
