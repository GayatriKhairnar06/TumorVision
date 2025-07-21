import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Configuration (MUST MATCH YOUR TRAINING CONFIG) ---
MODEL_PATH = './mobilenet_v2_best.keras' # <--- IMPORTANT: Update this
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary'] # <--- CONFIRM THIS ORDER

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

model = load_model()

if model:
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=["jpg", "jpeg", "png"]
    )
    st.markdown("---")
st.subheader("🧠 Tumor Types Information")

st.markdown("""
#### 1. **Glioma Tumor**
- Originates in glial cells.
- Often malignant and aggressive.
- Symptoms: Seizures, headaches, personality changes.

#### 2. **Meningioma Tumor**
- Develops in the meninges (protective brain lining).
- Mostly benign but can grow large.
- Symptoms: Headaches, vision issues, coordination problems.

#### 3. **Pituitary Tumor**
- Found in the pituitary gland (controls hormones).
- Usually benign.
- Symptoms: Hormonal imbalances, fatigue, vision changes.

#### 4. **No Tumor**
- MRI shows normal brain tissue.
""")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded MRI Image', use_container_width=True)
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
            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, all_predictions)):
                st.write(f"- {class_name.replace('_', ' ').title()}: **{prob*100:.2f}%**")

            os.remove(temp_file_path)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Please ensure the uploaded file is a valid image (JPG, JPEG, PNG).")
    
else:
    st.warning("Model could not be loaded. Please ensure `MODEL_PATH` is correct and the model file exists.")
