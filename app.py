from flask import Flask, render_template, request
from tensorflow import keras
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = keras.models.load_model("Predict.h5")

@app.before_request
def before_request():
    # Reset result and image_path at the beginning of each request
    request.result = None
    request.image_path = None

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img.reshape(1, 256, 256, 1)  # Reshape for model input
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', result=result, image_path=request.image_path, error="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', result=result, image_path=request.image_path, error="No selected file")

        try:
            img_path = "uploads/" + file.filename
            file.save(img_path)
            img = preprocess_image(img_path)
            result = model.predict(img)
            result = np.argmax(result)

            if result == 0:
                result = "Normal"
            elif result == 1:
                result = "Covid"
            else:
                result = "Viral Pneumonia"

            # Set the image path in the request context
            request.image_path = os.path.abspath(img_path)

        except Exception as e:
            print(e)
            return render_template('index.html', result=result, image_path=request.image_path, error="Error processing image")

    return render_template('index.html', result=result, image_path=request.image_path, error=None)

if __name__ == '__main__':
    app.run(debug=True)
