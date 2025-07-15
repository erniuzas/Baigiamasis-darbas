from services.data_loader import load_images_from_folder, prepare_data, load_test_data_with_labels
from models.neural.cnn_model import cnn_with_new_hyperparameters
from services.trainer import train_model
from visualizations.training_plots import plot_training_results
import numpy as np
import os
import random
import tensorflow as tf
from database.init_db import init_db
# from models.classical.Randomforest import RandomForestModel

init_db()  # Initialize the database

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

def main():
    set_seeds(42)

    # 1. Parametrai
    train_image_folder = 'data/training'
    test_image_folder = 'data/test/Images'
    test_csv_path = 'data/test/GT-final_test.csv'

    if os.path.exists(test_csv_path):
        print(f"Failas rastas: {test_csv_path}")
    else:
        print(f"Failas nerastas: {test_csv_path}")
        return

    input_shape = (64, 64, 1)

    # 2. Duomenų paruošimas
    X_train, y_train = load_images_from_folder(train_image_folder, grayscale=True)
    X_train, X_val, y_train, y_val, X_test, y_test = prepare_data(
        train_image_folder, test_folder=test_image_folder, test_csv=test_csv_path)

    num_classes = len(np.unique(y_train))

    print(f"Testavimo duomenų kiekis: {len(X_test)}")
    print(f"Klasių kiekis testavimui: {num_classes}")
    print(f'X_train size: {X_train.shape[0]}')
    print(f'y_train size: {y_train.shape[0]}')

    # 3. CNN modelis
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

    plot_training_results(history_custom, labels=("Custom CNN"))

    # 4. Random Forest (jei norėsi)
    # rf_model = RandomForestModel()
    # rf_model.fit(X_train, y_train)
    # rf_model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()