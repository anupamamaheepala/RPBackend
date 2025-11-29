from fastapi import APIRouter
from services.db_service import get_db
from models.user import UserData

router = APIRouter(prefix="/data", tags=["data"])

@router.get("/test")
def test_db_connection():
    db = get_db()
    collections = db.list_collection_names()
    return {"status": "success", "collections": collections}

@router.post("/add-user")
async def add_user(user: UserData):
    db = get_db()
    db.users.insert_one({"username": user.username, "email": user.email})
    return {"message": "User data saved successfully", "username": user.username, "email": user.email}