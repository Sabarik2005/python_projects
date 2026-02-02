from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load MobileNetV2 (Pre-trained on ImageNet)
model = MobileNetV2(weights='imagenet')

# Supported formats check
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'avif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_bytes):
    # PIL handles WebP and AVIF automatically
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'label': 'No file part', 'confidence': '0%'})
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'label': 'Invalid file type', 'confidence': '0%'})

        img_bytes = file.read()
        processed_img = prepare_image(img_bytes)
        
        preds = model.predict(processed_img)
        # Decode the top result
        results = decode_predictions(preds, top=1)[0]
        
        animal_name = results[0][1].replace('_', ' ').title()
        confidence = float(results[0][2] * 100)
        
        return jsonify({
            'label': animal_name,
            'confidence': f"{confidence:.2f}%"
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'label': 'Error processing image', 'confidence': '0%'})

if __name__ == '__main__':
    app.run(debug=True)