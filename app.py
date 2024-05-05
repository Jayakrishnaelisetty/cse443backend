import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model 
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications import ResNet50
import joblib

# Loading the saved model using joblib
def load_saved_model(filename):
    return joblib.load(filename)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_array, axis=0)
    features = base_model.predict(img_array)
    features_flattened = features.flatten()
    return features_flattened

def predict_from_model(model_filename, img_path):
    img_array = cv2.imread(img_path)
    features = extract_features(img_array)
    model = load_saved_model(model_filename)
    result = model.predict([features]) # Assuming features is a list
    if result == 1:
        return "Recyclable"
    elif result == 0:
        return "Organic"

# Allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def main():
    st.title('Image Classifier')

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        model_filename = 'decision_tree_model (2).pkl'

        # Save the uploaded file to a temporary location
        temp_dir = 'uploads'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        predicted_class_index = predict_from_model(model_filename, temp_filepath)
        
        st.write("Prediction:", predicted_class_index)

        # Remove the temporary uploaded file
        os.remove(temp_filepath)

if __name__ == '__main__':
    main()
