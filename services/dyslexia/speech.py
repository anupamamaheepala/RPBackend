# services/dyslexia/speech.py

import openai

def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=f
        )
    return transcript["text"]