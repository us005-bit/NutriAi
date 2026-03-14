"""
NutriAI - ocr.py
==============
OCR for mess menu board photos.

Primary  : Google Cloud Vision API (no binary, handles bad lighting + handwriting)
Fallback : pytesseract (if Vision API key not set or call fails)

Pipeline:
    Image bytes
        → Google Vision API (or Tesseract fallback)
        → raw text lines
        → clean + filter ignore words
        → fuzzy match against nutrition_db keys
        → enrich with nutrition data + natural serving units
        → return matched dishes

Natural serving units:
    Instead of raw grams, dishes get human-friendly units:
    dal/curry/soup   → "1 cup"
    rice/khichdi     → "1 cup (cooked)"
    roti/paratha     → "1 piece"
    idli/vada/dosa   → "2 pieces" / "1 piece"
    egg              → "1 egg"
    default fallback → serving_desc from nutrition_db

Install:
    pip install google-cloud-vision Pillow rapidfuzz --break-system-packages
    # For Tesseract fallback only:
    pip install pytesseract --break-system-packages

Environment variables (.env):
    GOOGLE_VISION_API_KEY=AIza_xxxx    # get from Google Cloud Console

Usage:
    from ocr import extract_menu_dishes
    result = extract_menu_dishes(image_bytes)
"""

import io
import os
import re
import base64
import logging
from typing import Optional

import httpx
from rapidfuzz import process, fuzz
from nutrition_db import NUTRITION_DB, build_result, get_all_keys
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("nutriai.ocr")

# ── Constants ──────────────────────────────────────────────────────────────────
OCR_FUZZY_THRESHOLD = 65   # minimum fuzzy score to accept a dish match

# Words to filter out from OCR lines — not food items
_IGNORE_WORDS = {
    "menu", "today", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "breakfast", "lunch", "dinner",
    "special", "items", "mess", "canteen", "hostel", "college",
    "am", "pm", "time", "morning", "evening", "night",
    "rs", "inr", "price", "free", "notice", "board",
    "dal", "sabji", "sabzi",   # too generic — kept as filters
}

# ── Natural serving unit rules ─────────────────────────────────────────────────
# Maps keyword patterns in dish names to natural serving descriptions.
# Checked in order — first match wins.
_SERVING_UNIT_RULES = [
    # Liquids / semi-liquids
    (["soup", "rasam", "sambhar", "sambar", "kadhi", "chaas", "lassi",
      "buttermilk", "tea", "coffee", "milk", "juice"],          "1 cup"),

    # Dal / curry / sabzi / gravy
    (["dal", "daal", "curry", "masala", "sabzi", "sabji", "palak",
      "chana", "chole", "rajma", "matar", "paneer", "korma",
      "gravy", "fry", "tadka"],                                  "1 cup"),

    # Rice dishes
    (["rice", "pulao", "biryani", "khichdi", "khichri", "fried_rice",
      "jeera_rice", "peas_rice"],                                "1 cup (cooked)"),

    # Flatbreads — per piece
    (["roti", "chapati", "paratha", "naan", "puri", "bhatura",
      "thepla", "kulcha"],                                       "1 piece"),

    # South Indian — per piece
    (["dosa", "uttapam", "appam"],                               "1 piece"),
    (["idli"],                                                    "2 pieces"),
    (["vada", "medu"],                                           "2 pieces"),

    # Eggs
    (["egg", "anda", "omelette", "boiled_egg"],                  "1 egg"),

    # Snacks — per piece or small plate
    (["samosa", "kachori", "pakora", "bhajiya", "cutlet",
      "tikki", "patty", "sandwich", "burger"],                   "1 piece"),
    (["momos", "momo"],                                          "6 pieces"),
    (["poha", "upma", "sheera", "halwa", "daliya"],              "1 bowl"),

    # Sweets / desserts
    (["kheer", "payasam", "pudding", "custard"],                 "1 bowl"),
    (["ladoo", "laddoo", "barfi", "halwa", "rasgulla",
      "gulab_jamun"],                                             "1 piece"),

    # Salad / raita / curd
    (["raita", "curd", "dahi", "salad", "sprouts"],              "1 cup"),
]


def _get_natural_serving(dish_key: str, db_entry: dict) -> str:
    """
    Return a natural serving description for a dish.

    Checks _SERVING_UNIT_RULES against the dish key first.
    Falls back to serving_desc from nutrition_db if no rule matches.
    Falls back to portion_g if serving_desc is also missing.
    """
    dk = dish_key.lower()

    for keywords, unit in _SERVING_UNIT_RULES:
        if any(kw in dk for kw in keywords):
            return unit

    # Use nutrition_db serving_desc if available
    if db_entry.get("serving_desc"):
        return db_entry["serving_desc"]

    # Last resort — show grams
    pg = db_entry.get("portion_g", 0)
    return f"{int(pg)}g" if pg else "1 serving"


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING (shared by both backends)
# ══════════════════════════════════════════════════════════════════════════════
def _preprocess_image(image_bytes: bytes):
    """
    Greyscale + upscale to improve OCR accuracy.
    Applied before both Vision API and Tesseract.
    """
    from PIL import Image, ImageFilter, ImageOps
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) < 1000:
        scale = 1000 / max(w, h)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageOps.grayscale(img)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def _image_to_bytes(pil_img) -> bytes:
    """Convert PIL image back to bytes for API upload."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY: GOOGLE CLOUD VISION API
# ══════════════════════════════════════════════════════════════════════════════
def _vision_api_ocr(image_bytes: bytes) -> Optional[list[str]]:
    """
    Call Google Cloud Vision API TEXT_DETECTION.
    Returns list of raw text lines, or None if API call fails.

    Why Vision API over Tesseract:
    - No binary to install — works on any cloud platform
    - Handles bad lighting, chalk boards, handwritten menus
    - Free tier: 1000 calls/month — more than enough for one-time setup
    - Much better accuracy on real-world mess menu photos
    """
    api_key = os.getenv("GOOGLE_VISION_API_KEY")
    if not api_key:
        log.warning("GOOGLE_VISION_API_KEY not set — falling back to Tesseract")
        return None

    # Encode image as base64
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "requests": [{
            "image"   : {"content": img_b64},
            "features": [{"type": "TEXT_DETECTION", "maxResults": 1}],
        }]
    }

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    try:
        resp = httpx.post(url, json=payload, timeout=20)
        resp.raise_for_status()

        data        = resp.json()
        annotations = data["responses"][0].get("textAnnotations", [])

        if not annotations:
            log.warning("Vision API returned no text annotations")
            return []

        # First annotation contains the full text block
        full_text = annotations[0].get("description", "")
        raw_lines = [line.strip() for line in full_text.splitlines() if line.strip()]

        log.info(f"Google Vision API extracted {len(raw_lines)} raw lines ✅")
        return raw_lines

    except httpx.HTTPStatusError as e:
        log.error(f"Vision API HTTP error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        log.error(f"Vision API call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK: TESSERACT
# ══════════════════════════════════════════════════════════════════════════════
def _tesseract_ocr(image_bytes: bytes) -> list[str]:
    """
    Tesseract OCR fallback.
    Used when GOOGLE_VISION_API_KEY is not set or Vision API call fails.
    Requires tesseract binary installed on the system.
    """
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except ImportError:
        raise RuntimeError(
            "pytesseract not installed. Run: pip install pytesseract --break-system-packages"
        )
    except Exception:
        raise RuntimeError(
            "Tesseract binary not found.\n"
            "Ubuntu: sudo apt-get install tesseract-ocr\n"
            "macOS : brew install tesseract\n"
            "Or set GOOGLE_VISION_API_KEY in .env for cloud OCR (recommended)."
        )

    img      = _preprocess_image(image_bytes)
    raw_text = pytesseract.image_to_string(img, config="--psm 6 --oem 3 -l eng")
    raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    log.info(f"Tesseract extracted {len(raw_lines)} raw lines")
    return raw_lines


# ══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def _clean_line(raw: str) -> Optional[str]:
    """
    Clean a raw OCR line for fuzzy matching.
    Returns None if line should be discarded.
    """
    s = raw.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # remove special chars
    s = re.sub(r"\s+", " ", s).strip()

    if len(s) < 3:
        return None
    if re.fullmatch(r"[\d\s]+", s):       # pure numbers (prices, times)
        return None
    if s in _IGNORE_WORDS:
        return None
    # Skip lines that are only ignore words
    words = set(s.split())
    if words.issubset(_IGNORE_WORDS):
        return None
    return s


def _normalize_for_match(text: str) -> str:
    """Normalize cleaned line to nutrition_db key format (underscores)."""
    return re.sub(r"\s+", "_", text.strip())


# ══════════════════════════════════════════════════════════════════════════════
# FUZZY MATCHING + ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════
def _fuzzy_match_dish(text: str) -> Optional[dict]:
    """
    Fuzzy match a cleaned OCR line against nutrition_db keys.
    Returns enriched dish dict with nutrition + natural serving unit.
    Returns None if no match above threshold.
    """
    normalized = _normalize_for_match(text)
    table_keys = get_all_keys()

    match, score, _ = process.extractOne(
        normalized, table_keys, scorer=fuzz.token_sort_ratio
    )

    if score < OCR_FUZZY_THRESHOLD:
        log.debug(f"OCR fuzzy: '{normalized}' → '{match}' score={score:.0f} REJECTED")
        return None

    log.info(f"OCR fuzzy: '{normalized}' → '{match}' score={score:.0f} OK")

    # Build result from nutrition_db
    result = build_result(match)

    # Get db entry for serving unit lookup
    db_entry = NUTRITION_DB.get(match, {})

    # Override serving_desc with natural unit
    result["serving_desc"]    = _get_natural_serving(match, db_entry)
    result["ocr_confidence"]  = round(score, 1)
    result["ocr_raw_text"]    = text

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def extract_menu_dishes(image_bytes: bytes) -> dict:
    """
    Run OCR on a mess menu board photo and return matched dishes.

    Pipeline:
        1. Try Google Vision API (fast, accurate, no binary)
        2. Fallback to Tesseract if Vision API unavailable
        3. Clean + filter raw lines
        4. Fuzzy match against nutrition_db
        5. Enrich with nutrition data + natural serving units

    Args:
        image_bytes: Raw image bytes from file upload

    Returns:
        {
            "matched"    : [ dish dicts with nutrition + natural serving ],
            "raw_lines"  : [ all raw OCR lines ],
            "unmatched"  : [ lines that didn't match any dish ],
            "total_found": int,
            "ocr_backend": "google_vision" | "tesseract"
        }
    """
    log.info("Starting OCR pipeline...")
    ocr_backend = "google_vision"

    # Preprocess image for Vision API
    img        = _preprocess_image(image_bytes)
    img_bytes  = _image_to_bytes(img)

    # Try Google Vision API first
    raw_lines = _vision_api_ocr(img_bytes)

    if raw_lines is None:
        # Vision API failed or key not set — use Tesseract
        log.info("Falling back to Tesseract OCR...")
        ocr_backend = "tesseract"
        raw_lines   = _tesseract_ocr(image_bytes)

    log.info(f"OCR backend: {ocr_backend} | Raw lines: {len(raw_lines)}")

    # Clean + fuzzy match
    matched     = []
    unmatched   = []
    seen_dishes = set()

    for line in raw_lines:
        cleaned = _clean_line(line)
        if not cleaned:
            continue

        dish_result = _fuzzy_match_dish(cleaned)

        if dish_result and dish_result["dish"] not in seen_dishes:
            matched.append(dish_result)
            seen_dishes.add(dish_result["dish"])
        elif not dish_result:
            unmatched.append(line)

    log.info(f"OCR result: {len(matched)} matched, {len(unmatched)} unmatched")

    return {
        "matched"    : matched,
        "raw_lines"  : raw_lines,
        "unmatched"  : unmatched,
        "total_found": len(matched),
        "ocr_backend": ocr_backend,
    }
