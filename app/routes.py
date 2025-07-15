from flask import Blueprint, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import cv2

main = Blueprint('main', __name__)

# Užkrauk išsaugotą modelį (arba naudok modelį iš `run.py`, jei įrašai į .h5)
model = load_model('custom_cnn_model.h5')

@main.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64)) / 255.0
            image = image.reshape(1, 64, 64, 1)

            pred = model.predict(image)
            prediction = f"Numatyta klasė: {np.argmax(pred)} (tikimybė: {np.max(pred):.2%})"

    return render_template("index.html", prediction=prediction)