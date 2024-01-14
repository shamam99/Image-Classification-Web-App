```markdown
# Image Classification Web App

## Overview

This project is a simple web application that allows users to upload an image and receive a classification result using a trained convolutional neural network (CNN). The CNN is trained to classify images into three categories: "Normal," "Covid," and "Viral Pneumonia."
The basic working version has been uploaded and the modified version has created a CNN and L2 model. You are free to use the version you wish.

## Project Structure

The project is structured as follows:

```
my_project/
|-- predict.h5
|-- uploads/
|-- app.py
|-- templates/
|   |-- index.html
|-- README.md
```

- **predict.h5:** The trained model file in the Hierarchical Data Format (HDF5) used by Keras.
- **uploads/:** A folder to store uploaded images.
- **app.py:** The Flask application script.
- **templates/:** HTML templates for rendering the web pages.

## Dependencies

- Python 3.11
- Flask
- OpenCV
- NumPy
- TensorFlow
- Matplotlib
- Seaborn

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app:**
   ```bash
   python app.py
   ```

   Access the application at [http://localhost:5000](http://localhost:5000).

## Usage

1. **Upload an image:**
   - Visit the web app and click on the "Upload and Predict" button.
   - Select an image file in .jpg, .jpeg, or .png format.

2. **View the result:**
   - The application will process the image and display the classification result.
   - The result includes the predicted class ("Normal," "Covid," or "Viral Pneumonia").

3. **View the uploaded image:**
   - Below the result, the uploaded image is displayed along with the prediction result.

## Customization

- **Model Training:**
  - To train a new model or fine-tune the existing one, refer to the training script and dataset.

- **HTML Styling:**
  - To customize the appearance of the web pages, modify the styles in the HTML files.

## Acknowledgments

- The CNN model used in this project is trained on the [Covid-19 Chest X-ray Dataset].
