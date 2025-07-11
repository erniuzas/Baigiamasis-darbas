from services.data_loader import load_images_from_folder, prepare_data
from models.neural.cnn_model import build_cnn_model, cnn_with_new_hyperparameters
from services.trainer import train_model
from visualizations.training_plots import plot_training_results
import numpy as np
from visualizations.show_image import show_grayscale_image



def main():
    # 1. Parametrai
    image_folder = 'data/training'
    input_shape = (64, 64, 1)

    # 2. Įkeliame ir paruošiame duomenis
    X_gray, y = load_images_from_folder(image_folder)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(image_folder)
    num_classes = len(np.unique(y))

    print(f"Iš viso paveikslėlių: {len(X_gray)}")
    print(f"Klasių kiekis: {num_classes}")
    print(f"Mokymui: {len(X_train)}, Validavimui: {len(X_val)}, Testavimui: {len(X_test)}")

    # 3. Originalus modelis
    model_default = build_cnn_model(input_shape, num_classes)
    history_default = train_model(model_default, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    loss_def, acc_def = model_default.evaluate(X_test, y_test, verbose=0)
    print(f"\n▶ Originalaus modelio testo tikslumas: {acc_def:.2%}")

    # 4. Patobulintas modelis su hiperparametrais
    model_custom = cnn_with_new_hyperparameters(
        input_shape=input_shape,
        num_classes=num_classes,
        filters=64,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        dense_units=256,
        dropout_rate=0.4,
        learning_rate=0.0005,
        use_batch_norm=True,
        l2_reg=0.001,
        optimizer_type='rmsprop'
    )
    history_custom = train_model(model_custom, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    loss_cust, acc_cust = model_custom.evaluate(X_test, y_test, verbose=0)
    print(f"▶ Patobulinto modelio testo tikslumas: {acc_cust:.2%}")

    plot_training_results(history_default, history_custom, labels=("Default", "Custom CNN"))

if __name__ == "__main__":
    main()