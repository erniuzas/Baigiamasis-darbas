from flask import Flask
import numpy as np
import tensorflow as tf
import cv2
import os

# Vaizdo apdorojimas — kaip treniravime
def preprocess_image(file):
    img_array = np.frombuffer(file.read(), np.uint8)
    img_color = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_color is None:
        raise ValueError("Netinkamas failas")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)
    img_resized = cv2.resize(img_blur, (64, 64))
    img_norm = img_resized / 255.0
    img_final = img_norm.astype(np.float32).reshape(1, 64, 64, 1)
    return img_final

