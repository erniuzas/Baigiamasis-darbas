import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_images_from_folder(folder):
    gray_images = []
    labels = []

    for root, _, files in os.walk(folder):
        class_id = os.path.basename(root)
        if class_id.isdigit():
            label = int(class_id)
            for filename in files:
                if filename.endswith('.ppm'):
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Konvertuojame į nespalvotą (grayscale)
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # Keičiame dydį į 64x64
                        resized_gray = cv2.resize(gray_img, (64, 64))
                        # Normalizuojame
                        normalized_gray = resized_gray / 255.0
                        gray_images.append(normalized_gray)
                        labels.append(label)

    return np.array(gray_images), np.array(labels)


def prepare_data(image_folder):
    X_gray, y = load_images_from_folder(image_folder)
    # Pakeičiame formą, kad tiktų modeliams
    X_gray = X_gray.reshape(X_gray.shape[0], 64, 64, 1)

    # Padalijame į train, val ir test rinkinius
    X_gray_temp, X_gray_test, y_temp, y_test = train_test_split(
        X_gray, y, test_size=0.2, random_state=42, stratify=y
    )

    X_gray_train, X_gray_val, y_train, y_val = train_test_split(
        X_gray_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    return X_gray_train, X_gray_val, X_gray_test, y_train, y_val, y_test