from sqlalchemy import create_engine
from database.models import Base


def init_db(db_url='sqlite:///traffic_signs.db'):
    """
    Initializes the database by creating tables based on the defined models.
    
    Args:
        db_url (str): The database URL. Default is SQLite in-memory database.
    """
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    print(f"Database initialized with URL: {db_url}")
    