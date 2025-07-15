import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Adjust path to include parent directory


from database.init_db import init_db
from database.operations import import_csv_to_db, fetch_signs, save_prediction

# 1. Sukuriam DB
init_db()

# 2. Įkeliam CSV duomenis
import_csv_to_db()

# 3. Paimam kelis įrašus ir atspausdinam
fetch_signs(limit=3)

# 4. Išsaugom pavyzdinę prognozę
save_prediction("example.png", 5, 0.92)