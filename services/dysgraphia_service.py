# services/dysgraphia_service.py
from services.db_service import get_db
from bson.binary import Binary
from datetime import datetime
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from scipy.spatial.distance import cdist
import json  # For loading templates

def save_dysgraphia_submission(submission_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save dysgraphia submission to MongoDB.
    
    Args:
        submission_data: Dict from Pydantic model (validated).
    
    Returns:
        Dict with 'ok' status and 'submission_id' on success, or 'error' on failure.
    """
    db = get_db()
    collection = db["dysgraphia_submissions"]
    
    try:
        # Add metadata
        doc = submission_data.dict()
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()
        
        # Insert and get ID
        result = collection.insert_one(doc)
        submission_id = str(result.inserted_id)
        
        # Analyze and store results
        analysis = analyze_dysgraphia_submission(submission_data)
        if analysis['ok']:
            # Update doc with analysis
            collection.update_one(
                {'_id': result.inserted_id},
                {'$set': {'analysis': analysis}}
            )
        
        return {
            "ok": True,
            "submission_id": submission_id,
            "message": "Submission saved and analyzed successfully",
            "analysis_summary": analysis  # Return to app for feedback
        }
    
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to save submission: {str(e)}"
        }

def get_dysgraphia_stats() -> Dict[str, Any]:
    """
    Optional: Get basic stats (e.g., total submissions by grade/activity).
    """
    db = get_db()
    collection = db["dysgraphia_submissions"]
    
    pipeline = [
        {"$group": {
            "_id": {"grade": "$grade", "activity_type": "$activity_type"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id.grade": 1}}
    ]
    
    stats = list(collection.aggregate(pipeline))
    return {"ok": True, "stats": stats}

# Load TrOCR model once (lazy load for efficiency)
_processor = None
_model = None
_templates = None  # For DTW templates
def _get_trocr_model():
    global _processor, _model
    if _model is None:
        try:
            _processor = TrOCRProcessor.from_pretrained("eshangj/TrOCR-Sinhala-finetuned")
            _model = VisionEncoderDecoderModel.from_pretrained("eshangj/TrOCR-Sinhala-finetuned")
        except Exception as e:
            print(f"Warning: Failed to load TrOCR model: {e}. Using dummy recognition.")
            _processor = None
            _model = None
    return _processor, _model

def _get_templates():
    global _templates
    if _templates is None:
        try:
            # Load from file (place templates.json in your project root)
            with open('templates.json', 'r', encoding='utf-8') as f:
                _templates = json.load(f)
        except FileNotFoundError:
            # Starter empty templates; populate as needed
            _templates = {}
            print("Warning: templates.json not found. DTW similarity will be 0.")
    return _templates

def render_strokes_to_image(strokes: List[Dict[str, List[Dict[str, float]]]], width: int = 128, height: int = 128) -> Image.Image:
    """
    Render strokes to a grayscale image (like handwriting on paper).
    Args: strokes from PromptData (list of {'points': [{'x': float, 'y': float}, ...]})
    Returns: PIL Image
    """
    img = Image.new('L', (width, height), 255)  # White background, grayscale
    draw = ImageDraw.Draw(img)
    
    # Collect all points for normalization
    all_points = [point for stroke in strokes for point in stroke['points']]
    if not all_points:
        return img
    
    xs = [p['x'] for p in all_points]
    ys = [p['y'] for p in all_points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        min_x, max_x = 0, 1
    if max_y == min_y:
        min_y, max_y = 0, 1
    
    def normalize_point(p: Dict[str, float]) -> Tuple[float, float]:
        norm_x = (p['x'] - min_x) / (max_x - min_x)
        norm_y = (p['y'] - min_y) / (max_y - min_y)
        return norm_x * (width - 10) + 5, norm_y * (height - 10) + 5  # Padding
    
    stroke_width = 2
    for stroke in strokes:
        points = stroke['points']
        if len(points) < 2:
            continue
        prev = normalize_point(points[0])
        for point in points[1:]:
            curr = normalize_point(point)
            draw.line([prev, curr], fill=0, width=stroke_width)  # Black ink
            prev = curr
    
    return img

def recognize_text_from_image(img: Image.Image) -> str:
    """
    Use TrOCR to recognize Sinhala text from rendered image.
    Returns: Recognized string (e.g., 'අ')
    """
    processor, model = _get_trocr_model()
    if processor is None or model is None:
        return ""  # Fallback if model load fails
    
    try:
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
    except Exception as e:
        print(f"Recognition error: {e}")
        return ""

def compute_stroke_similarity(user_strokes: List[Dict[str, List[Dict[str, float]]]], 
                              template_strokes: List[Dict[str, List[Dict[str, float]]]]) -> float:
    """
    Simple DTW-based similarity (0-1, higher = better match).
    Fallback if OCR fails. Uses predefined templates.
    """
    if not user_strokes or not template_strokes:
        return 0.0
    
    # Flatten to 1D sequences (x,y as features)
    def flatten_strokes(strokes):
        return [(p['x'], p['y']) for stroke in strokes for p in stroke['points']]
    
    user_seq = flatten_strokes(user_strokes)
    template_seq = flatten_strokes(template_strokes)
    
    if len(user_seq) == 0 or len(template_seq) == 0:
        return 0.0
    
    # DTW distance (lower = better)
    dist_matrix = cdist(user_seq, template_seq, metric='euclidean')
    n, m = dist_matrix.shape
    dtw_matrix = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_matrix[i-1][j-1]
            dtw_matrix[i][j] = cost + min(dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
    
    dtw_dist = dtw_matrix[n][m]
    max_dist = max(n, m) * 2  # Normalized scale (rough)
    similarity = max(0, 1 - (dtw_dist / max_dist))
    return similarity

def analyze_dysgraphia_submission(submission_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze prompts_data: Recognize text, compute match scores.
    Returns: {'ok': True, 'analyses': [per-prompt dicts with 'recognized_text', 'match_score', 'is_correct']}
    """
    analyses = []
    templates = _get_templates()
    
    for prompt_data in submission_data.get('prompts_data', []):
        prompt = prompt_data['prompt']
        strokes = prompt_data['strokes']  # List of {'points': [...]}
        
        # Render to image
        img = render_strokes_to_image(strokes)
        
        # Recognize
        recognized = recognize_text_from_image(img)
        
        # Simple string match (exact for now; use difflib.SequenceMatcher for fuzzy later)
        from difflib import SequenceMatcher  # Import here to avoid global
        match_score = SequenceMatcher(None, recognized, prompt).ratio()
        is_correct = match_score > 0.8  # Threshold (80% similarity)
        
        # Fallback DTW if low confidence
        if match_score < 0.5:
            template_strokes = templates.get(prompt, [])
            dtw_score = compute_stroke_similarity([strokes], [template_strokes]) if template_strokes else 0.0
            match_score = max(match_score, dtw_score)
            is_correct = match_score > 0.7
        
        analyses.append({
            'prompt': prompt,
            'recognized_text': recognized,
            'match_score': round(match_score, 2),
            'is_correct': is_correct,
            'time_taken': prompt_data.get('time_taken'),
        })
    
    overall_accuracy = sum(1 for a in analyses if a['is_correct']) / len(analyses) if analyses else 0
    return {
        'ok': True, 
        'analyses': analyses, 
        'overall_accuracy': round(overall_accuracy, 2)
    }