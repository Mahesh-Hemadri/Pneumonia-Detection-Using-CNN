# Pneumonia Detection using Transfer Learning (VGG16 & ResNet50)

This project implements a Pneumonia Detection system from chest X-ray images using deep learning transfer learning models: **VGG16** and **ResNet50**. It includes:

- Training with data augmentation on a publicly available dataset
- Saving and loading trained models
- GUI for image selection and prediction with **Tkinter**
- Confidence score output
- Grad-CAM visualization for model explainability

---

## Dataset

The dataset used is the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:

[https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

This dataset contains labeled chest X-ray images divided into train, test, and validation folders with categories: *Pneumonia* and *Normal*.

---

## Project Structure

- `vgg16_pneumonia.h5` - Trained VGG16 model weights  
- `resnet50_pneumonia.h5` - Trained ResNet50 model weights  
- `pneumonia_detection.py` - Main Python script (with GUI)  

---

## Requirements

- Python 3.8+  
- TensorFlow 2.x  
- numpy  
- matplotlib  
- OpenCV (`opencv-python`)  
- Pillow  
- Tkinter (usually pre-installed with Python)  

Install dependencies with:

```bash
pip install tensorflow numpy matplotlib opencv-python pillow

