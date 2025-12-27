# routes/dyslexia_routes.py

from fastapi import APIRouter
import random
from services.dyslexia_loader import load_docx_sentences

router = APIRouter()

@router.get("/get-random")
def get_random_sentence(grade: int, level: int):
    try:
        sentences = load_docx_sentences(grade, level)
    except Exception as e:
        return {"error": str(e)}

    if not sentences:
        return {"error": "Document is empty"}

    chosen = random.choice(sentences)

    return {
        "grade": grade,
        "level": level,
        "sentence": chosen
    }