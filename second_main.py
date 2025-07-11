import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

image_folder = 'Training/'

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
                        # Keičiam dydį į 64x64
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        resized_gray = cv2.resize(gray_img, (64, 64))  # Pakeisti dydį į 64x64
                        normalized_gray = resized_gray / 255.0
                        gray_images.append(normalized_gray)

                        labels.append(label)

    return np.array(gray_images), np.array(labels)

# Įkeliame duomenis
X_gray, y = load_images_from_folder(image_folder)
X_gray = X_gray.reshape(X_gray.shape[0], 64, 64, 1)  # Keičiam formą, kad būtų tinkama CNN modeliams

# Padalijame į train, val ir test rinkinius
X_gray_train, X_gray_test, y_train, y_test = train_test_split(
    X_gray, y, test_size=0.2, random_state=42, stratify=y
)

X_gray_train, X_gray_val, y_train, y_val = train_test_split(
    X_gray_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# CNN modelio kūrimo funkcija
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Modelis
input_shape_gray = (64, 64, 1)  # Naujas įvesties dydis (64x64)
num_classes = len(np.unique(y))

model_gray = build_cnn_model(input_shape_gray, num_classes)

# Apmokymas su nurodyta validacijos aibe
history_gray = model_gray.fit(
    X_gray_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_gray_val, y_val)
)

# Įvertinimas
gray_loss, gray_acc = model_gray.evaluate(X_gray_test, y_test, verbose=0)

print(f"Nespalvoto modelio tikslumas: {gray_acc:.2f}")

# ========== 1. Nespalvoto (grayscale) modelio grafikai ==========

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history_gray.history['accuracy'], label='Train Accuracy', color='orange')
plt.plot(history_gray.history['val_accuracy'], label='Validation Accuracy', color='purple')
plt.title('Grayscale modelio tikslumas')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_gray.history['loss'], label='Train Loss', color='purple')
plt.plot(history_gray.history['val_loss'], label='Validation Loss', color='brown')
plt.title('Grayscale modelio praradimas')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.suptitle("Nespalvotų paveikslėlių modelio rezultatai", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print(f"Viso nespalvotų paveikslėlių: {len(X_gray)}")
print(f"Mokymui: {len(X_gray_train)}, Testavimui: {len(X_gray_test)}")