import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilename

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
BASE_DIR = 'chest_xray'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Data Generators
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

# Model Definitions
def create_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

def create_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    return model

# Training Function
def train_and_save_model(model, model_name):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS,
              steps_per_epoch=train_generator.samples // BATCH_SIZE,
              validation_steps=test_generator.samples // BATCH_SIZE)
    model.save(model_name)

# Train models if not saved
if not os.path.exists('vgg16_pneumonia.h5'):
    vgg16_model = create_vgg16_model()
    train_and_save_model(vgg16_model, 'vgg16_pneumonia.h5')

if not os.path.exists('resnet50_pneumonia.h5'):
    resnet50_model = create_resnet50_model()
    train_and_save_model(resnet50_model, 'resnet50_pneumonia.h5')

# Preprocess Input Image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Prediction Function with Confidence Score and Error Handling
def predict_pneumonia(image_path, model_path):
    try:
        model = load_model(model_path)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array)[0][0]
        label = "Pneumonia" if prediction > 0.5 else "Normal"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        confidence_percent = confidence * 100
        return label, confidence_percent
    except Exception as e:
        return None, str(e)

# Image Format Check
def check_image_format(image_path):
    try:
        img = load_img(image_path)
        if img.size != IMG_SIZE:
            messagebox.showwarning("Warning", "Image size is not 224x224")
        if img.mode != 'RGB':
            messagebox.showwarning("Warning", "Image is not in RGB format")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")
        return False
    return True

# Grad-CAM Utilities
def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(image_path, model_path, last_conv_layer_name):
    model = load_model(model_path)
    img_array = preprocess_image(image_path)
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)

    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Grad-CAM Visualization')
    plt.show()

# GUI for Image Selection and Prediction
def main():
    root = Tk()
    root.withdraw()  # Hide main window

    image_path = askopenfilename(title="Select a chest X-ray image",
                                 filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if not image_path:
        messagebox.showinfo("Info", "No image selected. Exiting.")
        return

    if not check_image_format(image_path):
        return

    # Predict with VGG16
    vgg_label, vgg_conf = predict_pneumonia(image_path, 'vgg16_pneumonia.h5')
    if vgg_label is None:
        messagebox.showerror("VGG16 Model Error", vgg_conf)
    else:
        messagebox.showinfo("VGG16 Prediction", f"Prediction: {vgg_label}\nConfidence: {vgg_conf:.2f}%")
        display_gradcam(image_path, 'vgg16_pneumonia.h5', last_conv_layer_name='block5_conv3')

    # Predict with ResNet50
    res_label, res_conf = predict_pneumonia(image_path, 'resnet50_pneumonia.h5')
    if res_label is None:
        messagebox.showerror("ResNet50 Model Error", res_conf)
    else:
        messagebox.showinfo("ResNet50 Prediction", f"Prediction: {res_label}\nConfidence: {res_conf:.2f}%")
        display_gradcam(image_path, 'resnet50_pneumonia.h5', last_conv_layer_name='conv5_block3_out')

if __name__ == "__main__":
    main()
