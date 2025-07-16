import cv2
import numpy as np

def preprocess_image(file):
    # Nuskaityti paveikslėlį į spalvotą formatą (nes JPG dažnai būna spalvotas)
    img_color = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    if img_color is None:
        raise ValueError("Neįmanoma atidaryti failo")

    # Konvertuojame į grayscale
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Naudojame histogramos sulyginimą – pagerina kontrastą
    img = cv2.equalizeHist(img)

    # (nebūtina, bet galima) – Pašalina smulkų JPG triukšmą
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Keičiam dydį
    img = cv2.resize(img, (64, 64))  

    # Normalizuojame
    img = img / 255.0
    img = img.astype(np.float32)

    # Į formą (1, 64, 64, 1)
    img = img.reshape(1, 64, 64, 1) 
    
    return img