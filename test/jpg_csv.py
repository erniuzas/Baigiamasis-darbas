import os
import csv
import pandas as pd

ppm_csv = 'data/test/GT-final_test.csv'
jpg_folder = 'data/test/Images_JPG/Images'
output_csv = 'data/test/GT-final_test_jpg.csv'

# Įkeliame originalų CSV
ppm_df = pd.read_csv(ppm_csv, sep=';')

# Sukuriame žemėlapį: '00000' -> ClassId
filename_to_class = {
    os.path.splitext(row['Filename'])[0]: row['ClassId']
    for _, row in ppm_df.iterrows()
}

# Renkame .jpg failus
jpg_files = sorted([
    f for f in os.listdir(jpg_folder)
    if f.lower().endswith('.jpg')
])

# Ruošiame duomenis
output_rows = []
for jpg_file in jpg_files:
    name = os.path.splitext(jpg_file)[0]  # be .jpg
    class_id = filename_to_class.get(name, None)
    if class_id is not None:
        output_rows.append([jpg_file, class_id])
    else:
        print(f"Nerasta klasė failui: {jpg_file} – paliekama 0")
        output_rows.append([jpg_file, 0])  # Jei neranda, default klasė

# Išsaugome naują CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['Filename', 'ClassId'])  # Header
    writer.writerows(output_rows)

print(f" Sukurtas CSV su klasėmis: {output_csv} ({len(output_rows)} įrašų)")