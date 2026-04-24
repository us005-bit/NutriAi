"""
FitAI - nutrition.py
====================
Nutrition lookup with layered fallback:
    1. Cache        (instant, persisted to disk)
    2. Exact match  (nutrition_db.py)
    3. Fuzzy match  (nutrition_db.py)
    4. Parallel LLM (HF Primary + HF Secondary simultaneously → consensus)
    5. Gemini       (last resort, heavily rate-limited on free tier)

Changes from v1:
    - Quantity-aware prompt: pass "2 cups of dal tadka" → LLM returns calories
      for THAT exact quantity, not a generic "standard serving"
    - Divergence handling: if two LLMs disagree by >40% on calories, retry once.
      If still diverging, trust the primary (Qwen-72B) over secondary.
    - TTL on low-confidence cache entries: re-queried after 7 days instead of
      being served stale forever.

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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv  # ADD THIS
load_dotenv()
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
FUZZY_THRESHOLD       = 78
LLM_MAX_RETRIES       = 1       # Gemini only — fails fast
LLM_RETRY_DELAY       = 5
DIVERGENCE_THRESHOLD  = 40.0    # % calorie difference above which we retry
LOW_CONF_TTL_DAYS     = 7       # low-confidence cache entries expire after 7 days

# Gemini — last resort only
_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

# HuggingFace — both parallel workers use the same API, different models
_HF_API_URL     = "https://router.huggingface.co/v1/chat/completions"

_HF_PRIMARY_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",      # replace 32B — it's 400ing
]

_HF_SECONDARY_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",  # replace Mixtral — it's deprecated
    "meta-llama/Llama-3.2-3B-Instruct",
]

# ── Quantity-aware prompt ─────────────────────────────────────────────────────
# {quantity} is injected at call time, e.g. "1 ladle (~150g)" or "2 cups"
# If no quantity is specified, defaults to "1 standard serving"
_NUTRITION_PROMPT = (
    "You are a nutrition database. "
    "Return ONLY a JSON object for: {quantity} of {dish}\n\n"
    "Required keys: dish_name_corrected, calories, protein, carbs, fats, portion_g\n"
    "Rules:\n"
    "- All values must be numbers (int or float), never strings or null\n"
    "- calories/protein/carbs/fats: for the EXACT quantity specified above, not per 100g\n"
    "- portion_g: actual weight in grams of the quantity specified\n"
    '- If not a real food: {{"error": "not_a_food"}}\n'
    "- Return ONLY valid JSON. No explanation, no markdown, no backticks."
)

# Default quantity label shown in serving_desc when caller doesn't specify
_DEFAULT_QUANTITY = "1 standard serving"


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


def _cache_key(dish_name: str, quantity: str) -> str:
    """
    Cache key includes quantity so "1 ladle of dal" and "2 ladles of dal"
    are stored separately. Normalise quantity to keep keys clean.
    """
    norm_dish = normalize(dish_name)
    norm_qty  = re.sub(r"[^a-z0-9]", "_", quantity.lower().strip()).strip("_")
    return f"{norm_dish}__{norm_qty}"


# ══════════════════════════════════════════════════════════════════════════════
# CACHE  (with TTL support for low-confidence entries)
# ══════════════════════════════════════════════════════════════════════════════
def _load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        log.info(f"Cache loaded ({len(data)} entries)")
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

def _is_cache_valid(entry: dict) -> bool:
    """
    Returns False if a low-confidence entry is older than LOW_CONF_TTL_DAYS.
    High-confidence entries never expire (they came from DB or consensus).
    """
    if entry.get("confidence") != "low":
        return True
    cached_at = entry.get("cached_at")
    if not cached_at:
        return False   # no timestamp → treat as stale
    age = datetime.now() - datetime.fromisoformat(cached_at)
    if age > timedelta(days=LOW_CONF_TTL_DAYS):
        log.info(f"Low-confidence cache entry expired ({age.days} days old) — will re-query")
        return False
    return True


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
        "carbs"    : round(float(data["carbs"]),      1),
        "fats"     : round(float(data["fats"]),       1),
        "portion_g": round(float(data["portion_g"]),  0),
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
    return key

def _setup_huggingface() -> Optional[str]:
    key = os.getenv("HF_API_KEY")
    if not key:
        log.warning("HF_API_KEY not set — HuggingFace disabled.")
        return None
    return key


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL LLM CALLERS
# ══════════════════════════════════════════════════════════════════════════════
def _call_hf_model_list(
    api_key    : str,
    dish_name  : str,
    quantity   : str,
    models     : list,
    worker_name: str,
) -> Optional[dict]:
    """Call HuggingFace with a prioritised model list, trying each on failure."""
    prompt  = _NUTRITION_PROMPT.format(dish=dish_name, quantity=quantity)
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


def _call_gemini(api_key: str, dish_name: str, quantity: str) -> Optional[dict]:
    prompt  = _NUTRITION_PROMPT.format(dish=dish_name, quantity=quantity)
    payload = {
        "contents"        : [{"parts": [{"text": prompt}]}],
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
# CONSENSUS — divergence detection, retry, fallback to primary
# ══════════════════════════════════════════════════════════════════════════════
def _calorie_divergence_pct(r1: dict, r2: dict) -> float:
    """Percent difference in calories between two results."""
    max_cal = max(r1["calories"], r2["calories"], 1)
    return abs(r1["calories"] - r2["calories"]) / max_cal * 100


def _average_results(r1: dict, r2: dict) -> dict:
    """
    Average numeric nutrition fields from two LLM results.
    Should only be called when divergence is within acceptable range.
    """
    fields   = ["calories", "protein", "carbs", "fats", "portion_g"]
    averaged = {f: round((r1[f] + r2[f]) / 2, 1) for f in fields}
    diff_pct = _calorie_divergence_pct(r1, r2)
    confidence = "high" if diff_pct < 20 else "medium"

    log.info(
        f"Consensus: primary={r1['calories']}kcal  secondary={r2['calories']}kcal  "
        f"→ avg={averaged['calories']}kcal  divergence={diff_pct:.1f}%  conf={confidence}"
    )
    return {
        "dish"      : r1["dish"],
        "calories"  : averaged["calories"],
        "protein"   : averaged["protein"],
        "carbs"     : averaged["carbs"],
        "fats"      : averaged["fats"],
        "portion_g" : averaged["portion_g"],
        "source"    : "llm_consensus",
        "confidence": confidence,
        "cached_at" : datetime.now().isoformat(),
    }


def _run_parallel_once(dish_name: str, quantity: str) -> tuple[Optional[dict], Optional[dict]]:
    """Run both HF workers simultaneously. Returns (primary_result, secondary_result)."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_primary   = executor.submit(
            _call_hf_model_list, _hf_api_key, dish_name, quantity, _HF_PRIMARY_MODELS, "HF-Primary"
        )
        f_secondary = executor.submit(
            _call_hf_model_list, _hf_api_key, dish_name, quantity, _HF_SECONDARY_MODELS, "HF-Secondary"
        )
        try:
            r_primary = f_primary.result(timeout=35)
        except Exception as e:
            log.error(f"HF-Primary raised: {e}")
            r_primary = None
        try:
            r_secondary = f_secondary.result(timeout=35)
        except Exception as e:
            log.error(f"HF-Secondary raised: {e}")
            r_secondary = None

    return r_primary, r_secondary


def _query_parallel(dish_name: str, quantity: str) -> Optional[dict]:
    """
    Run two HuggingFace workers simultaneously.

    Flow:
        1. Both succeed + divergence ≤ 40%  → average  (high/medium confidence)
        2. Both succeed + divergence > 40%  → RETRY ONCE
             Retry both succeed + still diverging  → trust primary (Qwen-72B)
             Retry resolves              → average retry results
        3. Only one succeeds             → use it as-is
        4. Both fail                     → return None
    """
    if not _hf_api_key:
        return None

    r_primary, r_secondary = _run_parallel_once(dish_name, quantity)

    if r_primary and r_secondary:
        diff = _calorie_divergence_pct(r_primary, r_secondary)
        log.info(f"First run: primary={r_primary['calories']} secondary={r_secondary['calories']} diff={diff:.1f}%")

        if diff <= DIVERGENCE_THRESHOLD:
            # Happy path — models agree, safe to average
            return _average_results(r_primary, r_secondary)

        # ── Diverged — retry both once ────────────────────────────────────────
        log.warning(
            f"Divergence {diff:.1f}% exceeds {DIVERGENCE_THRESHOLD}% threshold — retrying both workers"
        )
        r2_primary, r2_secondary = _run_parallel_once(dish_name, quantity)

        if r2_primary and r2_secondary:
            diff2 = _calorie_divergence_pct(r2_primary, r2_secondary)
            log.info(f"Retry: primary={r2_primary['calories']} secondary={r2_secondary['calories']} diff={diff2:.1f}%")

            if diff2 <= DIVERGENCE_THRESHOLD:
                # Retry resolved the disagreement
                return _average_results(r2_primary, r2_secondary)

            # Still diverging after retry — trust primary (Qwen-72B is more reliable)
            log.warning(
                f"Still diverging after retry ({diff2:.1f}%) — "
                f"trusting primary ({r2_primary['calories']} kcal) over secondary ({r2_secondary['calories']} kcal)"
            )
            result = r2_primary.copy()
            result["source"]     = "llm_primary_diverged"
            result["confidence"] = "low"
            result["cached_at"]  = datetime.now().isoformat()
            return result

        # Retry had a partial failure — use whichever came back
        if r2_primary:
            r2_primary["confidence"] = "low"
            r2_primary["cached_at"]  = datetime.now().isoformat()
            return r2_primary
        if r2_secondary:
            r2_secondary["confidence"] = "low"
            r2_secondary["cached_at"]  = datetime.now().isoformat()
            return r2_secondary

        # Both retry workers failed — fall back to first run's primary
        log.warning("Retry workers both failed — using original primary despite divergence")
        r_primary["source"]     = "llm_primary_diverged"
        r_primary["confidence"] = "low"
        r_primary["cached_at"]  = datetime.now().isoformat()
        return r_primary

    # Only one worker succeeded
    if r_primary:
        log.info("Only HF-Primary succeeded — using as-is.")
        r_primary["confidence"] = "medium"
        r_primary["cached_at"]  = datetime.now().isoformat()
        return r_primary
    if r_secondary:
        log.info("Only HF-Secondary succeeded — using as-is.")
        r_secondary["confidence"] = "medium"
        r_secondary["cached_at"]  = datetime.now().isoformat()
        return r_secondary

    log.warning("Both HF parallel workers failed.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE
# ══════════════════════════════════════════════════════════════════════════════
_gemini_api_key = _setup_gemini()
_hf_api_key     = _setup_huggingface()
_cache          = _load_cache()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def get_nutrition(dish_name: str, quantity: str = _DEFAULT_QUANTITY) -> dict:
    """
    Main entry point. Returns nutrition dict or raises RuntimeError.

    Args:
        dish_name : e.g. "Dal Tadka"
        quantity  : e.g. "1 ladle (~150g)", "2 cups", "half plate"
                    Defaults to "1 standard serving" if not provided.

    Lookup order:
        cache → exact DB → fuzzy DB → parallel(HF-Primary + HF-Secondary) → Gemini

    The quantity is passed into the LLM prompt so returned calories reflect
    the actual portion the user ate, not a generic reference value.

    NOTE: DB and fuzzy matches return per-standard-serving values and do NOT
    scale to the requested quantity. For precise scaling of DB entries, use
    scale_nutrition() after this call.
    """
    cache_key  = _cache_key(dish_name, quantity)
    normalized = normalize(dish_name)
    log.info(f"get_nutrition: '{dish_name}' qty='{quantity}' → key='{cache_key}'")

    # 1. Cache (with TTL check for low-confidence entries)
    if cache_key in _cache:
        entry = _cache[cache_key]
        if _is_cache_valid(entry):
            log.info(f"Cache hit: '{cache_key}' conf={entry.get('confidence', 'db')}")
            result = entry.copy()
            result["source"] = "cache"
            result["serving_desc"] = quantity
            return result
        else:
            log.info(f"Stale low-confidence cache entry for '{cache_key}' — removing")
            del _cache[cache_key]

    # 2. Exact DB match
    #    DB values are per-standard-serving. We return them as-is and let the
    #    caller use scale_nutrition() if they need a different quantity.
    if normalized in NUTRITION_DB:
        log.info(f"Exact DB match: '{normalized}'")
        result = build_result(normalized)
        result["serving_desc"] = "1 standard serving (DB)"
        result["confidence"]   = "db"
        _cache[cache_key] = result
        _save_cache(_cache)
        return result

    # 3. Fuzzy DB match
    fuzzy = _fuzzy_lookup(normalized)
    if fuzzy:
        log.info(f"Fuzzy DB match: '{normalized}' → '{fuzzy['dish']}'")
        fuzzy["serving_desc"] = "1 standard serving (DB fuzzy)"
        fuzzy["confidence"]   = "db"
        _cache[cache_key] = fuzzy
        _save_cache(_cache)
        return fuzzy

    # 4. Parallel HF Primary + HF Secondary
    log.info(f"Not in DB — running parallel dual-HF LLM for '{dish_name}' qty='{quantity}'")
    parallel_result = _query_parallel(dish_name, quantity)
    if parallel_result:
        parallel_result["serving_desc"] = quantity
        _cache[cache_key] = parallel_result
        _save_cache(_cache)
        return parallel_result
    log.warning("Parallel LLM failed — trying Gemini as last resort")

    # 5. Gemini — last resort
    if _gemini_api_key:
        log.info(f"Querying Gemini for '{dish_name}' qty='{quantity}'")
        gemini_result = _call_gemini(_gemini_api_key, dish_name, quantity)
        if gemini_result:
            gemini_result["serving_desc"] = quantity
            gemini_result["confidence"]   = gemini_result.get("confidence", "medium")
            gemini_result["cached_at"]    = datetime.now().isoformat()
            _cache[cache_key] = gemini_result
            _save_cache(_cache)
            return gemini_result
        log.warning("Gemini also failed.")

    raise RuntimeError(
        f"No nutrition data found for '{dish_name}'. "
        "Not in lookup table and all LLM providers failed."
    )


def get_nutrition_safe(dish_name: str, quantity: str = _DEFAULT_QUANTITY) -> dict:
    """Wrapper that never raises — returns error dict instead."""
    try:
        return get_nutrition(dish_name, quantity)
    except (ValueError, RuntimeError) as e:
        return {"error": True, "message": str(e), "dish": dish_name}


def scale_nutrition(result: dict, user_grams: float) -> dict:
    """
    Scale a nutrition result from its portion_g to user_grams.

    Useful when you have a DB entry (always per standard serving) but the user
    ate a different amount — e.g. DB says 150g, user ate 300g.

    Example:
        base = get_nutrition("Dal Tadka")          # 150g standard serving
        actual = scale_nutrition(base, 300)         # user ate 300g
    """
    if result.get("error"):
        return result
    base_g = result.get("portion_g", 0)
    if base_g <= 0:
        log.warning("scale_nutrition: portion_g is 0 — cannot scale")
        return result

    factor = user_grams / base_g
    scaled = result.copy()
    scaled["calories"]    = round(result["calories"]  * factor, 1)
    scaled["protein"]     = round(result["protein"]   * factor, 1)
    scaled["carbs"]       = round(result["carbs"]     * factor, 1)
    scaled["fats"]        = round(result["fats"]      * factor, 1)
    scaled["portion_g"]   = user_grams
    scaled["serving_desc"] = f"{user_grams}g (scaled)"
    return scaled


def prepopulate_cache(dish_list: Optional[list[str]] = None) -> dict:
    """Pre-fill cache for all known dishes. Zero LLM calls needed."""
    if dish_list is None:
        dish_list = get_all_keys()

    summary = {"success": [], "failed": []}
    print(f"\nPre-populating cache for {len(dish_list)} dishes...")
    print("-" * 55)

    for dish in dish_list:
        result = get_nutrition_safe(dish)    # uses default quantity
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
        ("Dal Tadka",               "1 ladle (~150g)"),
        ("del todka",               "1 standard serving"),
        ("Paneer Butter Masala",    "half plate"),
        ("Veg Momos",               "6 pieces"),
        ("Chicken Biryani",         "1 full plate"),
        ("Corn Flakes with Milk",   "1 bowl"),
        ("asfghj123",               "1 cup"),
        ("",                        "1 serving"),
    ]

    for dish, qty in tests:
        print(f"\nInput: '{dish}' | Qty: '{qty}'")
        r = get_nutrition_safe(dish, qty)
        if r.get("error"):
            print(f"  ERROR: {r['message']}")
        else:
            print(f"  {r['dish']} | {r['calories']} kcal | "
                  f"P:{r['protein']}g C:{r['carbs']}g F:{r['fats']}g | "
                  f"serving={r.get('serving_desc','?')} ({r['portion_g']}g) "
                  f"[{r['source']}] conf={r.get('confidence','db')}")
    import time

    # ── Timing benchmark ──────────────────────────────────────
print("\n" + "=" * 60)
print("Latency Benchmark")
print("=" * 60)

import time

dishes = ["Dal Tadka", "Paneer Butter Masala", "Rajma Chawal",
          "Chole Bhature", "Veg Biryani", "Chicken Biryani", "Poha"]

# Clear these dishes from cache to simulate a cold first lookup
for dish in dishes:
    key_default = _cache_key(dish, _DEFAULT_QUANTITY)
    if key_default in _cache:
        del _cache[key_default]

# Cold pass — hits DB (layer 2), writes to cache
times_cold = []
for dish in dishes:
    start = time.time()
    get_nutrition_safe(dish)
    times_cold.append(time.time() - start)

# Warm pass — pure cache hits
times_warm = []
for dish in dishes:
    start = time.time()
    get_nutrition_safe(dish)
    times_warm.append(time.time() - start)

print(f"Cold avg: {sum(times_cold)/len(times_cold)*1000:.1f}ms  (DB lookup + cache write)")
print(f"Warm avg: {sum(times_warm)/len(times_warm)*1000:.1f}ms  (cache hit)")
print(f"Reduction: {(1 - sum(times_warm)/sum(times_cold))*100:.0f}%")

# LLM latency test — use a dish NOT in your DB
print("\n── LLM Latency Test ──")
novel_dishes = ["Schezwan Egg Fried Rice", "Tandoori Mushroom Tikka", "Oats Upma"]

for dish in novel_dishes:
    # clear from cache if present
    key = _cache_key(dish, _DEFAULT_QUANTITY)
    if key in _cache:
        del _cache[key]
    
    start = time.time()
    result = get_nutrition_safe(dish)
    elapsed = (time.time() - start) * 1000
    
    if result.get("error"):
        print(f"  {dish}: FAILED ({elapsed:.0f}ms)")
    else:
        print(f"  {dish}: {elapsed:.0f}ms [{result['source']}] conf={result.get('confidence')}")