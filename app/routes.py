from flask import Blueprint, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from services.image_utils import preprocess_image  # <- tavo funkcija čia

main = Blueprint('main', __name__)
model = load_model('custom_cnn_model.h5')

@main.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            try:
                image = preprocess_image(file)
                pred = model.predict(image)
                prediction = f"Numatyta klasė: {np.argmax(pred)} (tikimybė: {np.max(pred):.2%})"
            except Exception as e:
                prediction = f"Klaida apdorojant vaizdą: {str(e)}"

    return render_template("index.html", prediction=prediction)