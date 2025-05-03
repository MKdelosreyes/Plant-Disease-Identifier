import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('PlantDiseaseClassifierModelV2.keras')

st.set_page_config(page_title="Plant Disease Identifier", layout="centered")

st.title("🌱 Plant Disease Identifier")
st.write("Upload an image of a plant leaf, and let the model predict the disease!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # predictions = model.predict(img_array)
    predicted_index = tf.argmax(prediction[0]).numpy()
    # predicted_class = image_classes[predicted_index]
    confidence = prediction[0][predicted_index]

    classes = ['Background_without_leaves', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 
               'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
               'Pepper,_bell___healthy', 'Potato___Late_blight', 'Potato___healthy', 
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
               'Tomato___Late_blight', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy']

    st.success(f"### 🧠 Prediction: **{classes[predicted_class[0]]} - {confidence:.2f}**")
