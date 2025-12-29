from fastapi import APIRouter, HTTPException, Body
from services.db_service import get_db
from models.user_model import UserCreate, UserLogin, UserUpdate
from passlib.context import CryptContext
from bson import ObjectId

router = APIRouter()
db = get_db()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- SIGNUP ---
@router.post("/auth/signup")
async def signup(user: UserCreate):
    # Check if user exists
    if db["users"].find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = pwd_context.hash(user.password)
    
    user_doc = {
        "username": user.username,
        "password": hashed_password,
        "age": user.age,
        "grade": user.grade
    }
    
    result = db["users"].insert_one(user_doc)
    return {"ok": True, "message": "User created successfully", "id": str(result.inserted_id)}

# --- LOGIN ---
@router.post("/auth/login")
async def login(user: UserLogin):
    user_doc = db["users"].find_one({"username": user.username})
    if not user_doc:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    if not pwd_context.verify(user.password, user_doc["password"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    return {
        "ok": True,
        "user_id": str(user_doc["_id"]),
        "username": user_doc["username"],
        "age": user_doc["age"],
        "grade": user_doc["grade"]
    }

# --- GET PROFILE ---
@router.get("/auth/profile/{user_id}")
async def get_profile(user_id: str):
    try:
        user_doc = db["users"].find_one({"_id": ObjectId(user_id)})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")
            
        return {
            "id": str(user_doc["_id"]),
            "username": user_doc["username"],
            "age": user_doc["age"],
            "grade": user_doc["grade"]
        }
    except:
         raise HTTPException(status_code=400, detail="Invalid User ID")

# --- UPDATE PROFILE ---
@router.put("/auth/profile/{user_id}")
async def update_profile(user_id: str, update_data: UserUpdate):
    try:
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
        
        if not update_dict:
            return {"ok": True, "message": "No changes made"}

        result = db["users"].update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_dict}
        )
        
        if result.modified_count == 0:
             raise HTTPException(status_code=404, detail="User not found or no changes made")

        return {"ok": True, "message": "Profile updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))