import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st


st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_indices.json"))

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Display Logo and Title
col1, col2 = st.columns([1, 5])
with col1:
    st.image(f"{working_dir}/logo.png", width=60)
with col2:
    st.title("Plant Disease Classifier")

st.markdown("Upload an image of a plant leaf to identify possible diseases.")

# Upload Section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((200, 200))
        st.image(resized_img, caption="Uploaded Leaf Image")

    with col2:
        if st.button('🔍 Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'🌱 Prediction: **{str(prediction)}**')
