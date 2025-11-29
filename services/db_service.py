from pymongo import MongoClient
from config.settings import settings

client = MongoClient(settings.MONGODB_URI)
db = client[settings.DB_NAME]

def get_db():
    try:
        client.admin.command("ping")
        print("Connected to MongoDB!")
        return db
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise