from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight
import tensorflow as tf
import numpy as np

def augment_dataset(X, y):
    augmented_images = []
    augmented_labels = []

    for img, label in zip(X, y):
        img_aug = tf.image.random_flip_left_right(img)
        img_aug = tf.image.random_brightness(img_aug, max_delta=0.1)
        img_aug = tf.image.random_contrast(img_aug, lower=0.9, upper=1.1)
        augmented_images.append(img_aug.numpy())
        augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=20, batch_size=64,
                callbacks=None, use_augmentation=True):
    
    assert X_train.shape[0] == y_train.shape[0], "X_train ir y_train turi sutapti pagal pavyzdžių kiekį"

    # Augmentacija
    if use_augmentation:
        X_aug, y_aug = augment_dataset(X_train, y_train)
        X_train = np.concatenate([X_train, X_aug])
        y_train = np.concatenate([y_train, y_aug])

    # Callback'ai
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

    if callbacks is None:
        callbacks = [early_stop, reduce_lr, model_checkpoint]

    # Klasės svoriai
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Modelio treniravimas
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )

    return history