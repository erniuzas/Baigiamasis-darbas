def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    assert X_train.shape[0] == y_train.shape[0], f"X_train ({X_train.shape[0]}) ir y_train ({y_train.shape[0]}) dydžiai nesutampa!"
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )
    return history