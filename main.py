from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
import os
from PIL import Image

# Define the path to the training directory
train_dir = "chest_xray/train"

# List all subdirectories (categories) in the training directory
categories = os.listdir(train_dir)

# Create empty lists to store images and labels
images = []
labels = []

# Loop through each category (subdirectory)
for category in categories:
    category_dir = os.path.join(train_dir, category)
    
    # Loop through each image file in the category directory
    for image_name in os.listdir(category_dir):
        # Load the image
        image_path = os.path.join(category_dir, image_name)
        image = Image.open(image_path)
        # Resize the image to the target size
        resized_image = image.resize(IMAGE_SIZE)
        image_np = np.array(image)
        # Convert the resized image to a numpy array and flatten it
        flattened_image = np.array(image_np).flatten()
        # Resize the image if necessary                
        
        # Append the image to the list of images
        images.append(flattened_image)
        
        # Append the label (category) to the list of labels
        labels.append(category)
# Define the path to the training directory
val_dir = "chest_xray/val"

# List all subdirectories (categories) in the training directory
categories = os.listdir(val_dir)

# Create empty lists to store images and labels
images_val = []
labels_val = []

# Loop through each category (subdirectory)
for category in categories:
    category_dir = os.path.join(val_dir, category)
    
    # Loop through each image file in the category directory
    for image_name in os.listdir(category_dir):
        # Load the image
        image_path = os.path.join(category_dir, image_name)
        image = Image.open(image_path)
        resized_image = image.resize(IMAGE_SIZE)
        image_np = np.array(image)
        # Convert the resized image to a numpy array and flatten it
        flattened_image = np.array(image_np).flatten()
        
        # Append the image to the list of images
        images_val.append(flattened_image)
        
        # Append the label (category) to the list of labels
        labels_val.append(category)

# Create empty lists to store images and labels
images_test = []
labels_test = []
test_dir = "chest_xray/test"

# Loop through each category (subdirectory)
for category in categories:
    category_dir = os.path.join(test_dir, category)
    
    # Loop through each image file in the category directory
    for image_name in os.listdir(category_dir):
        # Load the image
        image_path = os.path.join(category_dir, image_name)
        image = Image.open(image_path)
        
        resized_image = image.resize(IMAGE_SIZE)
        image_np = np.array(image)
        # Convert the resized image to a numpy array and flatten it
        flattened_image = np.array(image_np).flatten()
        
        # Append the image to the list of images
        images_test.append(flattened_image)
        
        # Append the label (category) to the list of labels
        labels_test.append(category)
# Define data generators for train, validation, and test sets
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# train_generator = train_datagen.flow_from_directory(
#     directory='chest_xray/train',
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='training'
# )

# val_generator = train_datagen.flow_from_directory(
#     directory='chest_xray/train',
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='validation'
# )

# test_datagen = ImageDataGenerator(rescale=1./255)

# test_generator = test_datagen.flow_from_directory(
#     directory='chest_xray/test',
#     target_size=IMAGE_SIZE,
#     batch_size=1,
#     class_mode=None,
#     shuffle=False
# )
# Flatten the images and convert them into a single numpy array

# Convert the list of labels into a numpy array
labels_array = np.array(labels)

# Create and train the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed
knn_classifier.fit(images, labels_array)

# Predict on the validation set
y_val_pred = knn_classifier.predict(images_val)

# Calculate accuracy on the validation set
val_accuracy = accuracy_score(labels_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Evaluate on the test set
y_test_pred = knn_classifier.predict(images_test)
test_accuracy = accuracy_score(labels_test, y_test_pred)
print("Test Accuracy:", test_accuracy)