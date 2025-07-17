from services.data_loader import load_images_from_folder, load_test_data_with_labels, load_data
from models.neural.cnn_model import cnn_with_new_hyperparameters
from services.trainer import train_model
from visualizations.training_plots import plot_training_results
import numpy as np
import os
import random
import tensorflow as tf
from database.init_db import init_db
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Inicializuojame duomenų bazę
init_db()

def main():
    #  Duomenų keliai
    train_image_folder = 'data/training/JPG'
    jpg_test_folder = 'data/test/Images_JPG/Images'
    jpg_csv = 'data/test/GT-final_test_jpg.csv'

    if not os.path.exists(jpg_csv):
        print(f" Nerastas failas: {jpg_csv}")
        return

    input_shape = (64, 64, 1)

    # 🔹 Duomenų paruošimas (automatiškai normalizuojami viduje load_data)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        train_image_folder,
        jpg_test_folder,
        jpg_csv
    )

    num_classes = len(np.unique(y_train))
    print(f" Testavimo duomenų kiekis: {len(X_test)}")
    print(f" Klasių kiekis testavimui: {num_classes}")

    # Klasių pasiskirstymas treniravimo rinkinyje
    unique_train, train_counts = np.unique(y_train, return_counts=True)
    print(" Treniruotės klasių pasiskirstymas:")
    for cls, count in zip(unique_train, train_counts):
        print(f"  Klasė {cls}: {count} pavyzdžių")

    #  CNN modelis su pasirinktais hiperparametrais
    model_custom = cnn_with_new_hyperparameters(
    input_shape=X_train.shape[1:],
    num_classes=num_classes,
    learning_rate=0.0001
)
    # Callback'ai (EarlyStopping jau turimas)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    #  Modelio treniravimas su augmentacija
    history_custom = train_model(
        model=model_custom,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=20,
        batch_size=64,
        use_augmentation=True
    )

    #  Modelio išsaugojimas
    model_custom.save('cnn_model_jpg.keras')

    #  Įvertinimas testavimo rinkinyje
    loss_cust, acc_cust = model_custom.evaluate(X_test, y_test, verbose=0)
    print(f" Patobulinto modelio testo tikslumas: {acc_cust:.2%}")

    #  Prognozės
    y_pred = model_custom.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    #  Modelio spėjimų analizė
    unique_pred, pred_counts = np.unique(y_pred_classes, return_counts=True)
    print(" Modelio spėjimų pasiskirstymas:")
    for cls, count in zip(unique_pred, pred_counts):
        print(f"  Spėta klasė {cls}: {count} kartų")

    #  Klasifikacijos ataskaita
    print("\n Klasifikacijos ataskaita:")
    print(classification_report(y_test, y_pred_classes, zero_division=0))

    #  Klaidingi pavyzdžiai
    errors = np.where(y_pred_classes != y_test)[0]
    print(f" Klaidingų klasifikacijų: {len(errors)}")

    for i in range(min(5, len(errors))):
        idx = errors[i]
        plt.imshow(X_test[idx].reshape(64, 64), cmap='gray')
        plt.title(f"Teisinga: {y_test[idx]}, Spėta: {y_pred_classes[idx]}")
        plt.axis('off')
        plt.show()

    #  Treniruotės grafikas
    plot_training_results(history_custom, labels=("Custom CNN"))

# Paleidimas
if __name__ == "__main__":
    main()