from app.database.database import Base, engine
from app.database.models import Memory


def init_database():
    Base.metadata.create_all(bind=engine)
    print("Recallix database initialized successfully.")


if __name__ == "__main__":
    init_database()