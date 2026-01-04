from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId

# Helper to handle MongoDB ObjectId
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

class UserCreate(BaseModel):
    username: str
    password: str
    age: int
    grade: int

class UserLogin(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    username: Optional[str] = None
    age: Optional[int] = None
    grade: Optional[int] = None
    avatar_image: Optional[str] = None  # <--- Added this field

class UserResponse(BaseModel):
    id: str
    username: str
    age: int
    grade: int
    avatar_image: Optional[str] = "plogo1" # <--- Added this field