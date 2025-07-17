from flask import Blueprint, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import base64
from services.image_utils import preprocess_image 

main = Blueprint('main', __name__)
model = load_model('cnn_model_jpg.keras')

label_map = {
    0: "Leistinas greitis (20 km/h)",
    1: "Leistinas greitis (30 km/h)",
    2: "Leistinas greitis (50 km/h)",
    3: "Leistinas greitis (60 km/h)",
    4: "Leistinas greitis (70 km/h)",
    5: "Leistinas greitis (80 km/h)",
    6: "Leistino greičio (80 km/h) pabaiga",
    7: "Leistinas greitis (100 km/h)",
    8: "Leistinas greitis (120 km/h)",
    9: "Lenkti draudžiama",
    10: "Lenkti draudžiama transporto priemonėms virš 3,5 t",
    11: "Pirmenybė sankryžoje",
    12: "Pagrindinis kelias",
    13: "Duoti kelią",
    14: "STOP ženklas",
    15: "Eismas draudžiamas",
    16: "Draudžiama transporto priemonėms virš 3,5 t",
    17: "Įvažiuoti draudžiama",
    18: "Bendras pavojus",
    19: "Pavojingas posūkis į kairę",
    20: "Pavojingas posūkis į dešinę",
    21: "Du posūkiai iš eilės",
    22: "Nelygus kelias",
    23: "Slidus kelias",
    24: "Kelias siaurėja iš dešinės",
    25: "Kelio darbai",
    26: "Šviesoforas",
    27: "Pėstieji",
    28: "Vaikų perėja",
    29: "Dviračių eismas",
    30: "Sniego ar ledo pavojus",
    31: "Laukiniai gyvūnai",
    32: "Visų apribojimų pabaiga",
    33: "Sukti dešinėn",
    34: "Sukti kairėn",
    35: "Tik tiesiai",
    36: "Tiesiai arba dešinėn",
    37: "Tiesiai arba kairėn",
    38: "Laikytis dešinės",
    39: "Laikytis kairės",
    40: "Privalomas apvažiavimas ratu",
    41: "Lenkimo draudimo pabaiga",
    42: "Lenkimo draudimo (virš 3,5 t) pabaiga"
}

@main.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None  # Naudojama rodyti atvaizduotą nuotrauką

    if request.method == "POST":
        file = request.files["image"]
        if file:
            try:
                # Pateikiame nuotrauką kaip base64 stringą, kad galėtume ją parodyti HTML
                image = preprocess_image(file)

                # Debugging: Patikrinkite apdoroto vaizdo formą
                print(f"Apdoroto vaizdo forma: {image.shape}")

                # Atidarome nuotrauką ir konvertuojame ją į base64
                img = Image.open(file.stream)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image_url = f"data:image/png;base64,{img_str}"

                # Modelio prognozė
                pred = model.predict(image)

                # Debugging: Atspausdinkite prognozes
                print(f"Modelio prognozės: {pred}")

                # Pasirinkti klasę pagal didžiausią tikimybę
                class_index = int(np.argmax(pred))
                class_name = label_map.get(class_index, "Nežinomas ženklas")
                confidence = float(np.max(pred))

                # Debugging - išvedame klasės indeksą ir tikimybę
                print(f"Modelis prognozavo: {class_index} -> {class_name}, tikimybė: {confidence:.2%}")
                
                # Pateikiame prognozę, bet tik jei tikimybė viršija tam tikrą slenkstį
                if confidence > 0.80:  # Slenkstis
                    prediction = f"Numatyta klasė: {class_name} (tikimybė: {confidence:.2%})"
                else:
                    prediction = f"Neaiški prognozė (tikimybė: {confidence:.2%})"
            
            except Exception as e:
                prediction = f"Klaida apdorojant vaizdą: {str(e)}"
                print("Klaida apdorojant vaizdą:", str(e))

    return render_template("index.html", prediction=prediction, image_url=image_url)