from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.applications import ResNet50
# Saving the trained model to a pickle file
import joblib

# Loading the saved model using joblib
def load_saved_model(filename):
    return joblib.load(filename) 

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def extract_features(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_array, axis=0)
    features = base_model.predict(img_array)
    features_flattened = features.flatten()
    return features_flattened

def predict_from_model(model_filename, img_path):
    # Assuming you have defined the necessary preprocessing functions and variables
    img_array = cv2.imread(img_path)
    features = extract_features(img_array)
    model = load_saved_model(model_filename)
    result = model.predict([features]) # Assuming features is a list
    # return prediction[0]
    
    if result == 1:
        return "Recyclable"
    elif result == 0:
        return "Organic"

# Allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        model_filename = 'decision_tree_model (2).pkl'
        predicted_class_index = predict_from_model(model_filename, image_path)
        print(predicted_class_index)
    return predicted_class_index

if __name__ == '__main__':    app.run(debug=True)
