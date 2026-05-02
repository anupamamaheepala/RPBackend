# models/request_models.py

from pydantic import BaseModel

class ReadingRequest(BaseModel):
    reference: str
    student: str