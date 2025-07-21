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
        st.error(f"Model not found at: {MODEL_PATH}")
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

# --- Streamlit UI Styling ---
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .main { background-color: #f8f5fc; }
        .block-container { padding-top: 2rem; }
        h1 { color: #6a1b9a; text-align: center; font-size: 40px; }
        h4, p { color: #444; }
        .uploadbox {
            background-color: #f3e5f5;
            padding: 30px;
            border-radius: 12px;
            border: 2px dashed #ba68c8;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 👩‍💻 About")
st.sidebar.markdown("**Gayatri Khairnar**  \n`AI Developer & Data Scientist`")
st.sidebar.markdown("📩 `khairnargayatri333@gmail.com`")
st.sidebar.markdown("> *Decoding the Universe, One Dataset at a Time*")

# Title and Description
st.markdown("<h1>🧠 Brain Tumor MRI Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a brain MRI image to receive an AI-powered classification of tumor type.</p>", unsafe_allow_html=True)

model = load_model()

if model:
    st.markdown("<div class='uploadbox'>📤 <strong>Upload MRI Image</strong> (JPG, JPEG, PNG)</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='🖼 Uploaded MRI Image', use_container_width=True)
        st.write("🧪 **Classifying...**")

        try:
            temp_file_path = "temp_image.png"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            img_array = preprocess_image(temp_file_path)
            predicted_class, confidence, all_predictions = predict_tumor_type(model, img_array, CLASS_NAMES)

            # Result Display
            st.markdown(f"""
                <div style="background-color:#e1bee7; padding:20px; border-radius:10px; text-align:center">
                    <h2 style="color:#4a148c;">Prediction: {predicted_class.replace('_', ' ').title()}</h2>
                    <p style="font-size:18px;">Confidence: <strong>{confidence:.2f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)

            st.subheader("🔍 All Class Probabilities")
            for class_name, prob in zip(CLASS_NAMES, all_predictions):
                st.write(f"- **{class_name.replace('_', ' ').title()}**: {prob*100:.2f}%")

            os.remove(temp_file_path)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.warning("Please ensure the uploaded file is a valid image (JPG, JPEG, PNG).")
else:
    st.warning("🚫 Model could not be loaded. Check the model path and file.")

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 13px; color: #aaa;'>
        © 2025 Gayatri Khairnar | Built with ❤️ using Streamlit
    </div>
""", unsafe_allow_html=True)
