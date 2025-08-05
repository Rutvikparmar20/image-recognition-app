from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model load કરો
model = tf.keras.applications.MobileNetV2(weights='imagenet')  # Pre-trained model

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filepath = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(filepath)

            # Image process કરો
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Prediction કરો
            preds = model.predict(img_array)
            decoded = decode_predictions(preds, top=1)[0][0]
            label = f"{decoded[1]} ({decoded[2]*100:.2f}%)"

            return render_template('index.html', label=label, image_path=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
# app.py - Flask application for image classification using MobileNetV2