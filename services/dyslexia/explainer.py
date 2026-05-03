# services/dyslexia/explainer.py

def get_char_type(char: str):
    """
    Identify Sinhala character type
    """
    vowels = ["ා", "ැ", "ෑ", "ි", "ී", "ු", "ූ", "ෙ", "ේ", "ො", "ෝ", "ෞ"]
    consonants = [
        "ක","ග","ච","ජ","ට","ඩ","ත","ද","ප","බ",
        "ම","න","ල","ව","ස","හ","ය","ර"
    ]

    if char in vowels:
        return "vowel"
    elif char in consonants:
        return "consonant"
    else:
        return "other"


def get_severity(error_type: str, char_type: str):
    """
    Assign severity level
    """
    if char_type == "vowel":
        return "medium"
    elif char_type == "consonant":
        return "high"
    else:
        return "low"


def generate_message(error_type: str, char: str, char_type: str):
    """
    Generate human-friendly explanation
    """

    # Sinhala vowel explanations
    vowel_explanations = {
        "ා": "long vowel 'aa' sound",
        "ි": "short vowel 'i' sound",
        "ී": "long vowel 'ee' sound",
        "ු": "short vowel 'u' sound",
        "ූ": "long vowel 'oo' sound",
        "ෙ": "short vowel 'e' sound",
        "ේ": "long vowel 'ee' sound",
    }

    if error_type == "missing":
        if char in vowel_explanations:
            return f"Missing {vowel_explanations[char]}"
        elif char_type == "consonant":
            return f"Missing consonant '{char}'"
        else:
            return f"Missing character '{char}'"

    elif error_type == "extra":
        return f"Extra character '{char}' detected"

    elif error_type == "replace":
        return f"Incorrect character, expected '{char}'"

    return "Reading inconsistency detected"


def generate_explanations(errors):
    """
    Convert raw errors into explainable feedback
    """

    explanations = []

    for err in errors:
        char = err.get("char", "")
        error_type = err.get("type", "unknown")
        position = err.get("position", -1)

        char_type = get_char_type(char)
        severity = get_severity(error_type, char_type)
        message = generate_message(error_type, char, char_type)

        explanations.append({
            "position": position,
            "type": error_type,
            "character": char,
            "message": message,
            "severity": severity
        })

    return explanations