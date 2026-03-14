"""
FitAI - nutrition.py
====================
Nutrition lookup with layered fallback:
    1. Cache        (instant, persisted to disk)
    2. Exact match  (nutrition_db.py)
    3. Fuzzy match  (nutrition_db.py)
    4. Parallel LLM (HF Primary + HF Secondary simultaneously → average result)
    5. Gemini       (last resort, heavily rate-limited on free tier)

Setup:
    pip install rapidfuzz httpx

    Add to .env:
        HF_API_KEY=hf_xxxx
        GEMINI_API_KEY=xxxx   (optional, last resort)
"""

import os
import re
import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from rapidfuzz import process, fuzz

from nutrition_db import NUTRITION_DB, build_result, get_all_keys


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("fitai.nutrition")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CACHE_FILE = BASE_DIR / "nutrition_cache.json"

# ── Constants ─────────────────────────────────────────────────────────────────
FUZZY_THRESHOLD = 78
LLM_MAX_RETRIES = 1       # Gemini only — fails fast
LLM_RETRY_DELAY = 5

# Gemini — last resort only
_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# HuggingFace — both parallel workers use the same API, different models
_HF_API_URL     = "https://router.huggingface.co/v1/chat/completions"

# Primary: Qwen 72B — best accuracy for nutrition/Indian food
_HF_PRIMARY_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",   # fallback if 72B is overloaded
]

# Secondary: Mistral / Llama — independent second opinion for averaging
_HF_SECONDARY_MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",  # small but always available
]

# Shared prompt template
_NUTRITION_PROMPT = (
    "You are a nutrition database. "
    "Return ONLY a JSON object for this dish: {dish}\n\n"
    "Required keys: dish_name_corrected, calories, protein, carbs, fats, portion_g\n"
    "Rules:\n"
    "- All values must be numbers (int or float), never strings or null\n"
    "- calories/protein/carbs/fats: per standard single serving (NOT per 100g)\n"
    "- portion_g: weight of standard serving in grams\n"
    '- If not a real food: {{"error": "not_a_food"}}\n'
    "- Return ONLY valid JSON. No explanation, no markdown, no backticks."
)


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════
def normalize(dish_name: str) -> str:
    if not dish_name or not dish_name.strip():
        raise ValueError("Dish name cannot be empty.")
    name = dish_name.lower().strip()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", "_", name)
    return name.strip("_")


# ══════════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════════
def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        log.info(f"Cache loaded ({len(data)} entries): {list(data.keys())}")
        return data
    log.info("No cache file — starting fresh.")
    return {}

def _save_cache(cache: dict) -> None:
    tmp = CACHE_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=2)
        tmp.replace(CACHE_FILE)
        log.info(f"Cache saved ({len(cache)} entries)")
    except Exception as e:
        log.error(f"Cache save failed: {e}")
        if tmp.exists():
            tmp.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# FUZZY MATCH
# ══════════════════════════════════════════════════════════════════════════════
def _fuzzy_lookup(normalized_name: str) -> Optional[dict]:
    table_keys = get_all_keys()
    match, score, _ = process.extractOne(
        normalized_name, table_keys, scorer=fuzz.token_sort_ratio,
    )
    log.info(f"Fuzzy: '{normalized_name}' → '{match}' (score: {score:.1f})")
    if score < FUZZY_THRESHOLD:
        log.warning(f"Score {score:.1f} below threshold — rejecting '{match}'")
        return None
    return build_result(match)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED JSON PARSER
# ══════════════════════════════════════════════════════════════════════════════
def _parse_llm_response(raw: str, dish_name: str, source: str) -> Optional[dict]:
    """Parse and validate JSON from any LLM response."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",           "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    # Extract JSON object (handles reasoning prefix from DeepSeek R1)
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON object found in {source} response")
    data = json.loads(json_match.group())

    if data.get("error") == "not_a_food":
        log.warning(f"{source}: '{dish_name}' is not a real food.")
        return None

    required = ["calories", "protein", "carbs", "fats", "portion_g"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing key: {key}")
        if not isinstance(data[key], (int, float)):
            raise ValueError(f"Non-numeric {key}: {data[key]}")

    if not (0 <= data["calories"] <= 3000):
        raise ValueError(f"Unrealistic calories: {data['calories']}")
    if not (0 <= data["protein"] <= 200):
        raise ValueError(f"Unrealistic protein: {data['protein']}")

    return {
        "dish"     : normalize(data.get("dish_name_corrected", dish_name)),
        "calories" : round(float(data["calories"]),  1),
        "protein"  : round(float(data["protein"]),   1),
        "carbs"    : round(float(data["carbs"]),     1),
        "fats"     : round(float(data["fats"]),      1),
        "portion_g": round(float(data["portion_g"]), 0),
        "source"   : source,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SETUP — API KEYS
# ══════════════════════════════════════════════════════════════════════════════
def _setup_gemini() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        log.warning("GEMINI_API_KEY not set — Gemini disabled.")
        return None
    log.info("Gemini API key loaded.")
    return key

def _setup_huggingface() -> Optional[str]:
    key = os.getenv("HF_API_KEY")
    if not key:
        log.warning("HF_API_KEY not set — HuggingFace disabled.")
        return None
    log.info("HuggingFace API key loaded.")
    return key


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL LLM CALLERS
# ══════════════════════════════════════════════════════════════════════════════
def _call_hf_model_list(api_key: str, dish_name: str, models: list, worker_name: str) -> Optional[dict]:
    """Call HuggingFace with a prioritized list of models, trying each on failure."""
    prompt  = _NUTRITION_PROMPT.format(dish=dish_name)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for model in models:
        payload = {
            "model"      : model,
            "messages"   : [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens" : 300,
        }
        try:
            resp = httpx.post(_HF_API_URL, json=payload, headers=headers, timeout=30)

            if resp.status_code == 503:
                log.warning(f"{worker_name} '{model}' loading — waiting 20s")
                time.sleep(20)
                resp = httpx.post(_HF_API_URL, json=payload, headers=headers, timeout=30)

            if resp.status_code in (429, 404, 410):
                log.warning(f"{worker_name} '{model}' returned {resp.status_code} — trying next")
                continue

            resp.raise_for_status()
            raw    = resp.json()["choices"][0]["message"]["content"].strip()
            result = _parse_llm_response(raw, dish_name, f"llm_{worker_name.lower()}")
            if result:
                log.info(f"{worker_name} success via '{model}'")
                return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.warning(f"{worker_name} '{model}' parse error: {e}")
        except httpx.HTTPStatusError as e:
            log.error(f"{worker_name} '{model}' HTTP error: {e}")
        except Exception as e:
            log.error(f"{worker_name} '{model}' error: {e}")

    log.error(f"{worker_name} — all models failed.")
    return None



def _call_gemini(api_key: str, dish_name: str) -> Optional[dict]:
    prompt = _NUTRITION_PROMPT.format(dish=dish_name)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }
    for model in _GEMINI_MODELS:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                resp = httpx.post(url, json=payload, timeout=30)

                if resp.status_code in (404, 429):
                    log.warning(f"Gemini '{model}' {resp.status_code} — skipping")
                    break

                resp.raise_for_status()
                raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                result = _parse_llm_response(raw, dish_name, "llm_gemini")
                if result:
                    log.info(f"Gemini success via '{model}'")
                    return result

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                log.warning(f"Gemini '{model}' attempt {attempt+1} parse error: {e}")
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(LLM_RETRY_DELAY)
            except Exception as e:
                log.error(f"Gemini '{model}' error: {e}")
                break

    log.error("Gemini — all models failed.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL DUAL-HF WITH AVERAGING
# ══════════════════════════════════════════════════════════════════════════════
def _average_results(r1: dict, r2: dict) -> dict:
    """Average numeric nutrition fields from two LLM results."""
    fields = ["calories", "protein", "carbs", "fats", "portion_g"]
    averaged = {f: round((r1[f] + r2[f]) / 2, 1) for f in fields}
    log.info(
        f"Consensus: primary={r1['calories']}kcal  secondary={r2['calories']}kcal  "
        f"→ avg={averaged['calories']}kcal"
    )
    return {
        "dish"     : r1["dish"],   # use primary model's corrected name
        "calories" : averaged["calories"],
        "protein"  : averaged["protein"],
        "carbs"    : averaged["carbs"],
        "fats"     : averaged["fats"],
        "portion_g": averaged["portion_g"],
        "source"   : "llm_consensus",
    }


def _query_parallel(dish_name: str) -> Optional[dict]:
    """
    Run two HuggingFace workers simultaneously in threads.
    - Both succeed  → average their results  (source: llm_consensus)
    - One succeeds  → use that one as-is
    - Both fail     → return None
    """
    if not _hf_api_key:
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_primary   = executor.submit(
            _call_hf_model_list, _hf_api_key, dish_name, _HF_PRIMARY_MODELS, "HF-Primary"
        )
        f_secondary = executor.submit(
            _call_hf_model_list, _hf_api_key, dish_name, _HF_SECONDARY_MODELS, "HF-Secondary"
        )
        try:
            r_primary   = f_primary.result(timeout=35)
        except Exception as e:
            log.error(f"HF-Primary raised: {e}")
            r_primary   = None
        try:
            r_secondary = f_secondary.result(timeout=35)
        except Exception as e:
            log.error(f"HF-Secondary raised: {e}")
            r_secondary = None

    if r_primary and r_secondary:
        log.info("Both HF workers succeeded — averaging results.")
        return _average_results(r_primary, r_secondary)
    if r_primary:
        log.info("Only HF-Primary succeeded — using as-is.")
        return r_primary
    if r_secondary:
        log.info("Only HF-Secondary succeeded — using as-is.")
        return r_secondary

    log.warning("Both HF parallel workers failed.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE
# ══════════════════════════════════════════════════════════════════════════════
_gemini_api_key   = _setup_gemini()
_hf_api_key       = _setup_huggingface()
_cache            = _load_cache()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def get_nutrition(dish_name: str) -> dict:
    """
    Main entry point. Returns nutrition dict or raises RuntimeError.

    Lookup order:
        cache → exact DB → fuzzy DB → parallel(HF-Primary + HF-Secondary) → Gemini
    """
    normalized = normalize(dish_name)
    log.info(f"get_nutrition: '{dish_name}' → '{normalized}'")

    # 1. Cache
    if normalized in _cache:
        log.info(f"Cache hit: '{normalized}'")
        result = _cache[normalized].copy()
        result["source"] = "cache"
        return result

    # 2. Exact DB match
    if normalized in NUTRITION_DB:
        log.info(f"Exact DB match: '{normalized}'")
        result = build_result(normalized)
        _cache[normalized] = result
        _save_cache(_cache)
        return result

    # 3. Fuzzy DB match
    fuzzy = _fuzzy_lookup(normalized)
    if fuzzy:
        log.info(f"Fuzzy DB match: '{normalized}' → '{fuzzy['dish']}'")
        _cache[normalized] = fuzzy
        _save_cache(_cache)
        return fuzzy

    # 4. Parallel HF Primary + HF Secondary (with averaging if both succeed)
    log.info(f"Not in DB — running parallel dual-HF LLM for '{dish_name}'")
    parallel_result = _query_parallel(dish_name)
    if parallel_result:
        _cache[normalized] = parallel_result
        _save_cache(_cache)
        return parallel_result
    log.warning("Parallel LLM failed — trying Gemini as last resort")

    # 5. Gemini — last resort
    if _gemini_api_key:
        log.info(f"Querying Gemini for '{dish_name}'")
        gemini_result = _call_gemini(_gemini_api_key, dish_name)
        if gemini_result:
            _cache[normalized] = gemini_result
            _save_cache(_cache)
            return gemini_result
        log.warning("Gemini also failed.")

    raise RuntimeError(
        f"No nutrition data found for '{dish_name}'. "
        "Not in lookup table and all LLM providers failed."
    )


def get_nutrition_safe(dish_name: str) -> dict:
    """Wrapper that never raises — returns error dict instead."""
    try:
        return get_nutrition(dish_name)
    except (ValueError, RuntimeError) as e:
        return {"error": True, "message": str(e), "dish": dish_name}


def prepopulate_cache(dish_list: Optional[list[str]] = None) -> dict:
    """Pre-fill cache for all known dishes. Zero LLM calls needed."""
    if dish_list is None:
        dish_list = get_all_keys()

    summary = {"success": [], "failed": []}
    print(f"\nPre-populating cache for {len(dish_list)} dishes...")
    print("-" * 55)

    for dish in dish_list:
        result = get_nutrition_safe(dish)
        if result.get("error"):
            print(f"  ✗ {dish:<35} FAILED: {result['message']}")
            summary["failed"].append(dish)
        else:
            print(f"  ✓ {dish:<35} {result['calories']:>6.0f} kcal  "
                  f"{result['protein']:>5.1f}g protein  [{result['source']}]")
            summary["success"].append(dish)

    print("-" * 55)
    print(f"Done — Success: {len(summary['success'])}  |  Failed: {len(summary['failed'])}")
    return summary


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("FitAI Nutrition Module — Quick Test")
    print("=" * 60)

    tests = [
        "Dal Tadka", "del todka", "Paneer Butter Masala",
        "Veg Momos", "Pani Puri", "Chicken Biryani",
        "Corn Flakes with Milk and Banana", "Waffle",
        "asfghj123", "",
    ]

    for dish in tests:
        print(f"\nInput: '{dish}'")
        r = get_nutrition_safe(dish)
        if r.get("error"):
            print(f"  ERROR: {r['message']}")
        else:
            print(f"  {r['dish']} | {r['calories']} kcal | "
                  f"P:{r['protein']}g C:{r['carbs']}g F:{r['fats']}g | "
                  f"({r['portion_g']}g) [{r['source']}]")