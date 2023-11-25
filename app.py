import streamlit as st
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import skimage.io

# Load the pre-trained VGG model
model = load_model("model_VGG.h5")

# Function to preprocess the image for the model
def preprocess_image(img_path):
    img = skimage.io.imread(img_path)
    img = skimage.transform.resize(img, (128, 128, 3))
    # img = image.load_img(img_path, target_size=(128, 128, 3))
    img_array = np.expand_dims(img, axis=0)
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    print("\n\n\n ===>",predictions)
    return predictions

# Streamlit app
st.title("Image Classifier Web App")

uploaded_file = st.file_uploader("Choose a file", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
        
    # Make a prediction
    predictions = predict(uploaded_file)

    # Display the top prediction
    class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']  # Update with your actual class names
    top_prediction = np.argmax(predictions)
    st.subheader("Prediction:")
    st.write(f"Class: {class_names[top_prediction]}, with probability: {predictions[0][top_prediction]}")

