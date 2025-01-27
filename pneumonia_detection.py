import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Define image size and paths
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
BASE_DIR = 'chest_xray'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescale validation/test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Function to create a model using VGG16
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False  # Freeze the layers of the base model
    return model

# Function to create a model using ResNet50
def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False  # Freeze the layers of the base model
    return model

# Function to train and save the model
def train_and_save_model(model, model_name):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS, steps_per_epoch=train_generator.samples // BATCH_SIZE, validation_steps=test_generator.samples // BATCH_SIZE)
    model.save(model_name)

# Check if models already exist, if not train and save them
if not os.path.exists('vgg16_pneumonia.h5'):
    vgg16_model = create_vgg16_model()
    train_and_save_model(vgg16_model, 'vgg16_pneumonia.h5')

if not os.path.exists('resnet50_pneumonia.h5'):
    resnet50_model = create_resnet50_model()
    train_and_save_model(resnet50_model, 'resnet50_pneumonia.h5')

# Function to preprocess the input image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict using VGG16 and ResNet50 models
def predict_pneumonia(image_path, model_path):
    model = load_model(model_path)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])  # Re-compile the model
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    return "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

# Edge case check for input image format
def check_image_format(image_path):
    try:
        img = load_img(image_path)
        if img.size != IMG_SIZE:
            print("Warning: Image size is not 224x224")
        if img.mode != 'RGB':
            print("Warning: Image is not in RGB format")
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True

# Create a Tkinter root window and hide it
Tk().withdraw()

# Ask the user to select an image file
image_path = askopenfilename(title="Select a chest X-ray image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])

if image_path:
    # Check image format
    if check_image_format(image_path):
        print("Image format check passed.")

        # Predict using VGG16
        vgg16_prediction = predict_pneumonia(image_path, 'vgg16_pneumonia.h5')
        print(f'VGG16 Prediction: {vgg16_prediction}')

        # Predict using ResNet50
        resnet50_prediction = predict_pneumonia(image_path, 'resnet50_pneumonia.h5')
        print(f'ResNet50 Prediction: {resnet50_prediction}')
    else:
        print("Image format check failed.")
else:
    print("No image selected.")
