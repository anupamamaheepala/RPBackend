from bson import ObjectId
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from config.settings import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_current_user():
    async def _get_current_user(token: str = Depends(oauth2_scheme)):
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=["HS256"]
            )

            username = payload.get("sub")
            user_id = payload.get("user_id")

            if not username or not user_id:
                raise HTTPException(status_code=401, detail="Invalid token")

            return {
                "username": username,
                "user_id": ObjectId(user_id)
            }

        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    return _get_current_user
