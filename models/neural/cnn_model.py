from tensorflow.keras import models, layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# # CNN modelio kūrimo funkcija
# def build_cnn_model(input_shape, num_classes):
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#         layers.MaxPooling2D(2, 2),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D(2, 2),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model


#funkcija kuri pakeicia modelio hyperparametrus
def cnn_with_new_hyperparameters(input_shape, num_classes, filters=32, kernel_size=(3, 3), pool_size=(2, 2), dense_units=128, 
                                 dropout_rate=0.5, learning_rate=0.001, use_batch_norm=True, l2_reg=0.01, optimizer_type='adam'):
    # Pasirinkti optimizatorių pagal naudotojo pasirinkimą
    if optimizer_type == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Nepalaikomas optimizatorius! Galimi variantai: 'adam', 'sgd', 'rmsprop'.")

    model = models.Sequential([
        layers.Input(shape=input_shape),  # Teisingas būdas nurodyti input_shape
        layers.Conv2D(filters, kernel_size, activation='relu', kernel_regularizer=l2(l2_reg)),
        layers.MaxPooling2D(pool_size),
        layers.Conv2D(filters * 2, kernel_size, activation='relu', kernel_regularizer=l2(l2_reg)),
        layers.MaxPooling2D(pool_size),
        
        # Jei norime naudoti batch normalization
        layers.BatchNormalization() if use_batch_norm else layers.Layer(),
        
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg)),
        
        # Dropout po tankio sluoksnio
        layers.Dropout(dropout_rate),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model