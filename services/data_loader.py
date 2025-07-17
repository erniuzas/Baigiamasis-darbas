from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import pandas as pd

def load_images_from_folder(folder, grayscale=True):
    images = []
    labels = []

    for root, _, files in os.walk(folder):
        class_id = os.path.basename(root)
        if class_id.isdigit():
            label = int(class_id)
            for filename in files:
                #  Paliekame tik JPG tipo failus
                if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg']:
                    img_path = os.path.join(root, filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Negalima atidaryti failo: {img_path}")
                        else:
                            if grayscale:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            img = cv2.resize(img, (64, 64))
                            img = img / 255.0
                            images.append(img.astype(np.float32))
                            labels.append(label)
                    except Exception as e:
                        print(f"Klaida apdorojant failą {img_path}: {e}")

    print(f" Įkelta vaizdų: {len(images)}")
    return np.array(images), np.array(labels, dtype=np.int32)

def load_test_data_with_labels(image_folder, labels_csv, grayscale=True):
    images = []
    labels = []

    df = pd.read_csv(labels_csv, sep=';')

    for i, row in df.iterrows():
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
            images.append(img.astype(np.float32))
            labels.append(label)

            # Tik pirmi keli testavimo pavyzdžiai
            if i < 5:
                print(f"{filename} -> klasė iš CSV: {row['ClassId']}, po korekcijos: {label}")

    print(f" Įkelta testavimo vaizdų: {len(images)}")
    return np.array(images), np.array(labels, dtype=np.int32)

# def load_combined_data(ppm_folder, jpg_folder, grayscale=True):
#     X_ppm, y_ppm = load_images_from_folder(ppm_folder, grayscale)
#     X_jpg, y_jpg = load_images_from_folder(jpg_folder, grayscale)
#     X_all = np.concatenate((X_ppm, X_jpg), axis=0)
#     y_all = np.concatenate((y_ppm, y_jpg), axis=0)
#     return X_all, y_all

def load_data(jpg_train_folder, jpg_test_folder, test_csv_jpg):
    #  Naudojame TIK JPG treniravimo duomenis
    X_all, y_all = load_images_from_folder(jpg_train_folder)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    #  Naudojame TIK JPG testavimo duomenis
    X_test, y_test = load_test_data_with_labels(jpg_test_folder, test_csv_jpg)

    # Performatuojame į (64, 64, 1)
    X_train = X_train.reshape(-1, 64, 64, 1)
    X_val = X_val.reshape(-1, 64, 64, 1)
    X_test = X_test.reshape(-1, 64, 64, 1)

    print(f" Final train size: {X_train.shape[0]}")
    print(f" Validation size: {X_val.shape[0]}")
    print(f" Test size: {X_test.shape[0]}")
    print("Test sample (first 10):", y_test[:10])

    return X_train, y_train, X_val, y_val, X_test, y_test