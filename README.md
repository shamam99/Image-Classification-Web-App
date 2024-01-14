```markdown
# Image Classification Web App

## Overview

This project is a simple web application that allows users to upload an image and receive a classification result using a trained convolutional neural network (CNN). The CNN is trained to classify images into three categories: "Normal," "Covid," and "Viral Pneumonia."
The basic working version has been uploaded and the modified version has created a CNN and L2 model. You are free to use the version you wish.


This code is an example of a Convolutional Neural Network (CNN) implemented using TensorFlow/Keras for image classification. Let's break down the main components of the code:

1. **Reading and Preprocessing Images:**
   - The code reads images from different directories, resizing them to a consistent size (256x256 pixels) and converting them to grayscale.
   - Images are loaded for the training set from directories like "Covid19-dataset/train/Normal", "Covid19-dataset/train/Covid", and "Covid19-dataset/train/Viral-Pneumonia".

2. **Creating Training and Testing Datasets:**
   - Images are split into training and testing datasets using `train_test_split` from scikit-learn.
   - The labels are converted to categorical using `tf.keras.utils.to_categorical`.

3. **Building the CNN Model:**
   - The model architecture is defined using Keras Sequential API.
   - Convolutional layers are used for feature extraction, and max-pooling layers downsample the spatial dimensions.
   - A dense layer with dropout is used for classification.
   - The output layer uses the softmax activation function for multi-class classification.

4. **Compiling and Training the Model:**
   - The model is compiled with the Adam optimizer and categorical crossentropy loss.
   - Training is performed using the `fit` method, and early stopping and learning rate reduction callbacks are applied.

5. **Evaluating the Model on Test Data:**
   - The trained model is evaluated on a separate set of images from the test dataset.

6. **Predicting a Single Image:**
   - An example image ("Covid19-dataset/test/Normal/shamamTest.jpeg") is loaded and preprocessed.
   - The model predicts the class of the image (Normal, Covid, or Viral-Pneumonia) and prints the result.

Regarding how the model classifies images:
- The CNN learns to recognize patterns and features in images. In the early layers, it captures simple features like edges and textures, while deeper layers learn to combine these features into more complex patterns.
- The model is trained to minimize the categorical crossentropy loss, which encourages it to assign high probabilities to the correct class.
- During training, the weights of the model are adjusted to improve its ability to discriminate between different classes.

Here are some considerations:

1. **Smaller Test Percentage (e.g., 0.1):**
   - **Advantages:**
     - More data available for training, which might be crucial if the dataset is small.
     - Faster training times.
   - **Disadvantages:**
     - The model may not generalize well to unseen data, especially if the test set is not representative of the overall data distribution.

2. **Moderate Test Percentage (e.g., 0.2 - 0.3):**
   - **Balanced Approach:**
     - A moderate test percentage strikes a balance between having enough data for training and ensuring a sufficiently large test set to evaluate the model's performance robustly.

3. **Larger Test Percentage (e.g., 0.3 and above):**
   - **Advantages:**
     - More reliable evaluation of the model's generalization performance.
     - Better identification of potential overfitting issues.
   - **Disadvantages:**
     - Reduced data for training, which might be a concern if the dataset is limited.

It's often recommended to start with a moderate test percentage (e.g., 0.2) and adjust based on the specific circumstances.

break down the components of the Convolutional Neural Network (CNN) model you've defined:

1. **Conv2D Layers:**
   - `layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 1))`: The first convolutional layer with 16 filters of size (3, 3), ReLU activation function, and input shape (256, 256, 1). This layer is responsible for detecting low-level features in the input image.

   - `layers.Conv2D(32, (3, 3), activation='relu')`: The second convolutional layer with 32 filters of size (3, 3) and ReLU activation function. This layer continues to extract more complex features.

   - `layers.Conv2D(64, (3, 3), activation='leaky_relu')`: The third convolutional layer with 64 filters of size (3, 3) and Leaky ReLU activation function. Leaky ReLU allows a small negative slope in the negative part of the activation.

   - `layers.Conv2D(128, (3, 3), activation='leaky_relu')`: The fourth convolutional layer with 128 filters of size (3, 3) and Leaky ReLU activation.

   - `layers.Conv2D(256, (3, 3), activation='leaky_relu')`: The fifth convolutional layer with 256 filters of size (3, 3) and Leaky ReLU activation.

2. **MaxPooling2D Layers:**
   - `layers.MaxPooling2D((2, 2))`: Max pooling layer with a pool size of (2, 2) after each convolutional layer. Max pooling reduces the spatial dimensions, preserving the most important information.

3. **Flatten Layer:**
   - `layers.Flatten()`: Flatten layer converts the 2D matrix data to a vector, preparing it for the fully connected layers.

4. **Dense Layers:**
   - `layers.Dense(256, activation='relu')`: Fully connected layer with 256 neurons and ReLU activation. This layer processes the flattened feature vector and captures high-level abstractions.

   - `layers.Dropout(0.5)`: Dropout layer with a dropout rate of 0.5. Dropout helps prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.

   - `layers.Dense(3, activation='softmax')`: Output layer with 3 neurons (assuming it's a classification task with three classes) and softmax activation. Softmax converts the network's raw output into probability scores for each class.

Overall, this CNN architecture follows a pattern of convolutional and pooling layers for feature extraction, followed by fully connected layers for classification. The use of Leaky ReLU, dropout, and softmax activation contributes to model robustness and effective training.


## Project Structure

The project is structured as follows:

## you should create a folder named uploads

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

- The CNN model used in this project is trained on the [Covid-19 Chest X-ray Dataset] URL: https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset.
