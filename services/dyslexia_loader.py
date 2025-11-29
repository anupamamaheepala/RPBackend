# services/dyslexia_loader.py

from docx import Document
import os

BASE_PATH = "data"

def get_docx_path(grade: int, level: int):
    # Example: data/grade3/Dyslexia/level2.docx
    return os.path.join(
        BASE_PATH,
        f"grade{grade}",
        "Dyslexia",
        f"level{level}.docx"
    )

def load_docx_sentences(grade: int, level: int):
    file_path = get_docx_path(grade, level)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist!")

    doc = Document(file_path)
    lines = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            lines.append(text)

    return lines
