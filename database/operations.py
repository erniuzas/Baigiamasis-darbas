import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from database.models import TrafficSigns, Predictions

def import_csv_to_db(csv_path='data/test/GT-final_test.csv', db_path='sqlite:///traffic_signs.db'):
    df = pd.read_csv(csv_path, sep=';')
    engine = create_engine(db_path)
    Session = sessionmaker(bind=engine)
    session = Session()

    for _, row in df.iterrows():
        sign = TrafficSigns(
            filename=row['Filename'],
            width=row['Width'],
            height=row['Height'],
            roi_x1=row['Roi.X1'],
            roi_y1=row['Roi.Y1'],
            roi_x2=row['Roi.X2'],
            roi_y2=row['Roi.Y2'],
            class_id=row['ClassId']
        )
        session.add(sign)

    session.commit()
    session.close()
    print("CSV įkeltas į duomenų bazę.")
def fetch_signs(limit=5):
    engine = create_engine('sqlite:///traffic_signs.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    signs = session.query(TrafficSigns).limit(limit).all()
    for sign in signs:
        print(f"{sign.filename} → klasė: {sign.class_id}")

    session.close()
    print("Duomenų bazė sukurta ir užpildyta iš CSV.")


def save_prediction(filename, predicted_class, confidence):
    engine = create_engine('sqlite:///traffic_signs.db')
    Session = sessionmaker(bind=engine)
    session = Session()

    result = Predictions(
        filename=filename,
        class_id=predicted_class,
        confidence=confidence
    )

    session.add(result)
    session.commit()
    session.close()