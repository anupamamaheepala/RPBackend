from dotenv import load_dotenv
import os
from pydantic_settings import BaseSettings

load_dotenv()  # loads .env from current working directory

class Settings(BaseSettings):
    MONGODB_URI: str
    DB_NAME: str
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Debug print to verify values (remove after testing)
print("MONGODB_URI:", settings.MONGODB_URI)
print("DB_NAME:", settings.DB_NAME)
print("OPENAI_API_KEY:", settings.OPENAI_API_KEY)
