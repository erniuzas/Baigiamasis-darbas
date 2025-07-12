import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder, grayscale=True):
    images = []
    labels = []

    for root, _, files in os.walk(folder):
        class_id = os.path.basename(root)
        if class_id.isdigit():
            label = int(class_id)
            for filename in files:
                if filename.endswith('.ppm'):
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Negalima atidaryti failo: {img_path}")
                    else:
                        if grayscale:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img, (64, 64))
                        img = img / 255.0  # normalizuojam į [0, 1]
                        images.append(img.astype(np.float32))  # užtikriname, kad būtų teisingas tipas
                        labels.append(label)

    print(f"✅ Įkelta treniravimo vaizdų: {len(images)}")
    return np.array(images), np.array(labels, dtype=np.int32)

def load_test_data_with_labels(image_folder, labels_csv, grayscale=True):
    images = []
    labels = []

    df = pd.read_csv(labels_csv, sep=';')

    for _, row in df.iterrows():
        filename = row['Filename']
        label = int(row['ClassId'])
        img_path = os.path.join(image_folder, filename)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Negalima atidaryti failo: {img_path}")
        else:
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            images.append(img.astype(np.float32))  # užtikriname, kad būtų teisingas tipas
            labels.append(label)

    print(f"✅ Įkelta testavimo vaizdų: {len(images)}")
    return np.array(images), np.array(labels, dtype=np.int32)

def load_data(train_folder, test_folder, test_csv):
    # Įkeliame treniravimo ir testavimo duomenis
    X_train, y_train = load_images_from_folder(train_folder)
    X_test, y_test = load_test_data_with_labels(test_folder, test_csv)

    # Pridedame kanalo dimensiją jei grayscale
    X_train = X_train.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)

    # Padaliname treniravimo duomenis į treniravimą ir validaciją
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    print(f"🔹 Final train size: {X_train.shape[0]}")
    print(f"🔹 Validation size: {X_val.shape[0]}")
    print(f"🔹 Test size: {X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test



def prepare_data(train_folder, test_folder=None, test_csv=None):
    # Įkeliame treniravimo duomenis
    X_train, y_train = load_images_from_folder(train_folder, grayscale=True)

    # Padaliname į mokymo ir validacijos duomenis
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    # Jei yra testavimo duomenys, įkeliame juos
    if test_folder and test_csv:
        X_test, y_test = load_test_data_with_labels(test_folder, test_csv)
        return X_train, X_val, y_train, y_val, X_test, y_test
    else:
        # Jei nėra testavimo duomenų, grąžiname tik treniravimo ir validacijos duomenis
        return X_train, X_val, y_train, y_val