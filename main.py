"""
NutriAI - main.py
================
FastAPI backend — wires all AI/ML modules into HTTP endpoints.

Run with:
    uvicorn main:app --reload --port 8000

Endpoints:

  HEALTH
    GET  /                          health check
    GET  /health                    model + device info

  FOOD CLASSIFIER
    POST /predict                   image → dish + nutrition
    POST /nutrition                 dish name → nutrition (5-layer)
    POST /suggest                   over-target → food swap suggestion

  OCR — MESS MENU SETUP (one-time)
    POST /ocr/scan                  photo → matched dishes + natural units
    POST /ocr/save-menu             save confirmed menu to PostgreSQL

  ONBOARDING AGENT
    POST /onboarding/start          start conversational onboarding
    POST /onboarding/reply          send reply → next question or plan
    GET  /onboarding/state/{id}     current session state
    DELETE /onboarding/{id}         clear session

  USER PROFILE
    GET  /user/{user_id}            fetch user profile + targets
    POST /user/{user_id}/update-weight  update current weight

  MEAL LOGGING (daily use)
    POST /log/meal                  log a meal item
    DELETE /log/{log_id}            delete a log entry
    GET  /log/today/{user_id}       today's logs
    GET  /log/totals/{user_id}      today's macro totals vs targets
    GET  /log/history/{user_id}     date-range log history

  GAP ANALYSIS AGENT
    POST /gap/start                 start gap analysis for weekly menu
    POST /gap/confirm               user confirms/skips a recommendation
    GET  /gap/status/{user_id}      current gap analysis state

  WEEKLY REVIEW
    GET  /review/{user_id}          7-day review — stats + insights + summary

  CHATBOT
    POST /chat                      send message → full response
    POST /chat/stream               send message → streaming response
    GET  /chat/history/{user_id}    fetch conversation history
    DELETE /chat/history/{user_id}  clear conversation

  RAG
    POST /rag/ask                   direct RAG nutrition Q&A
    POST /rag/populate              re-populate ChromaDB (admin)
"""

import json
import logging
import os
import re
from datetime import date
from typing import Optional, AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("nutriai.main")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "NutriAI API",
    description = "Agentic nutrition assistant for Indian college mess students",
    version     = "3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ══════════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup():
    log.info("Starting NutriAI backend v3...")

    # 1. Create PostgreSQL tables
    from database import create_tables
    await create_tables()

    # 2. Load EfficientNet-B2 classifier
    from model.classifier import get_classifier
    get_classifier()

    # 3. Pre-populate nutrition cache (all DB dishes → instant lookup)
    from nutrition import prepopulate_cache
    prepopulate_cache()

    # 4. Populate ChromaDB with nutrition_db (skips if already done)
    try:
        from rag import populate_nutrition_db
        populate_nutrition_db()
    except Exception as e:
        log.warning(f"RAG pre-population skipped: {e}")

    # 5. Initialize chatbot graph + PostgreSQL checkpointer
    try:
        from chatbot import get_graph
        await get_graph()
    except Exception as e:
        log.warning(f"Chatbot graph init skipped: {e}")

    log.info("NutriAI backend v3 ready ✅")


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

# ── Nutrition ─────────────────────────────────────────────────────────────────
class ManualEntryRequest(BaseModel):
    dish_name: str

class NutritionResult(BaseModel):
    dish       : str
    calories   : float
    protein    : float
    carbs      : float
    fats       : float
    portion_g  : float
    serving_desc: Optional[str] = None
    source     : str

class PredictionResponse(BaseModel):
    top_prediction  : dict
    all_predictions : list
    is_uncertain    : bool
    nutrition       : Optional[NutritionResult]
    nutrition_error : Optional[str]

class SuggestionRequest(BaseModel):
    calories_today  : float
    target_calories : float
    goal            : str
    protein_today   : float = 0
    target_protein  : float = 0

class SuggestionResponse(BaseModel):
    over_by          : float
    food_suggestion  : str
    food_calories    : int
    food_note        : str
    exercise         : str
    exercise_note    : str
    skipping_warning : str

# ── OCR ───────────────────────────────────────────────────────────────────────
class SaveMenuRequest(BaseModel):
    dishes        : list[dict]
    institution_id: str = "default"

class OCRMenuResponse(BaseModel):
    matched     : list
    raw_lines   : list
    unmatched   : list
    total_found : int
    ocr_backend : str

# ── Onboarding ────────────────────────────────────────────────────────────────
class OnboardingStartRequest(BaseModel):
    session_id: str
    user_id   : str

class OnboardingReplyRequest(BaseModel):
    session_id: str
    message   : str

class OnboardingResponse(BaseModel):
    session_id     : str
    question       : Optional[str]
    phase          : str
    done           : bool
    plan           : Optional[dict]
    profile_so_far : Optional[dict] = None
    error          : Optional[str]  = None

# ── Meal logging ──────────────────────────────────────────────────────────────
class LogMealRequest(BaseModel):
    user_id           : str
    meal_slot         : str       # breakfast | lunch | snacks | dinner
    dish_key          : str
    display_name      : str
    calories          : float
    protein           : float
    carbs             : float
    fats              : float
    serving_desc      : Optional[str]  = None
    portion_multiplier: float = 1.0   # 1.0 = standard, 2.0 = double
    source            : str   = "menu" # menu | scan | custom
    notes             : Optional[str]  = None
    log_date          : Optional[str]  = None  # YYYY-MM-DD, defaults to today

class DeleteLogRequest(BaseModel):
    user_id: str

# ── Gap analysis ──────────────────────────────────────────────────────────────
class GapAnalysisRequest(BaseModel):
    user_id    : str
    weekly_menu: dict   # { "monday": { "breakfast": [...], "lunch": [...] }, ... }

class GapConfirmRequest(BaseModel):
    user_id : str
    day     : str
    accepted: bool      # True = add to plan, False = skip

# ── Weekly review ─────────────────────────────────────────────────────────────
class WeeklyReviewResponse(BaseModel):
    stats   : dict
    insights: list
    summary : str

# ── Chatbot ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_id  : str
    message  : str
    thread_id: Optional[str] = None  # defaults to user_id

class RAGRequest(BaseModel):
    question   : str
    user_goal  : str  = "maintain"
    user_id    : Optional[str]  = None
    today_menu : Optional[dict] = None
    user_log   : Optional[dict] = None

class RAGResponse(BaseModel):
    question: str
    answer  : str
    sources : list


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message": "NutriAI backend v3 running 🍛", "status": "ok"}


@app.get("/health")
def health():
    from model.classifier import get_classifier
    c = get_classifier()
    return {
        "status" : "ok",
        "model"  : "EfficientNet-B2",
        "classes": len(c.classes),
        "device" : str(c.device),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FOOD CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Image upload → dish classification → nutrition lookup.
    Used in daily log tab when user scans food photo.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(400, f"Invalid file type: {file.content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file.")

    from model.classifier import get_classifier
    from nutrition import get_nutrition_safe

    classifier = get_classifier()
    prediction = classifier.predict(image_bytes, top_k=3)
    top_dish   = prediction["top_prediction"]["dish"]
    nutr       = get_nutrition_safe(top_dish)

    if nutr.get("error"):
        return PredictionResponse(
            **prediction,
            nutrition      = None,
            nutrition_error= nutr["message"],
        )

    return PredictionResponse(
        **prediction,
        nutrition      = NutritionResult(**nutr),
        nutrition_error= None,
    )


@app.post("/nutrition", response_model=NutritionResult)
def get_nutrition_endpoint(request: ManualEntryRequest):
    """
    Dish name → nutrition via 5-layer pipeline.
    Used when user types a custom food item in daily log.
    """
    if not request.dish_name.strip():
        raise HTTPException(400, "Dish name cannot be empty.")

    from nutrition import get_nutrition_safe
    result = get_nutrition_safe(request.dish_name)
    if result.get("error"):
        raise HTTPException(404, result["message"])
    return NutritionResult(**{
        k: result[k] for k in NutritionResult.model_fields if k in result
    })


@app.post("/suggest", response_model=SuggestionResponse)
def suggest_when_exceeded(request: SuggestionRequest):
    """
    User exceeded calorie target → suggest a light food + exercise.
    Never suggests skipping meals.
    """
    over_by = round(request.calories_today - request.target_calories, 1)
    if over_by <= 0:
        raise HTTPException(400, "Calories not exceeded yet.")

    import httpx
    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        return _static_suggestion(over_by)

    prompt = (
        f"College student ate {round(request.calories_today)} kcal "
        f"vs {round(request.target_calories)} target (goal: {request.goal}). "
        f"{round(over_by)} kcal over. "
        f"Protein: {round(request.protein_today)}g / {round(request.target_protein)}g.\n\n"
        "Suggest ONE light Indian mess food (under 120 kcal) AND one simple exercise. "
        "Never suggest skipping meals.\n\n"
        'Return ONLY JSON: food_suggestion, food_calories (int), food_note, '
        'exercise, exercise_note, skipping_warning.'
    )

    headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
    models  = ["Qwen/Qwen2.5-72B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1",
               "meta-llama/Llama-3.2-3B-Instruct"]

    for model in models:
        try:
            resp = httpx.post(
                "https://router.huggingface.co/v1/chat/completions",
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.3, "max_tokens": 350},
                headers=headers, timeout=30,
            )
            if resp.status_code != 200:
                continue
            raw  = resp.json()["choices"][0]["message"]["content"].strip()
            raw  = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw  = re.sub(r"\s*```$",           "", raw, flags=re.MULTILINE)
            m    = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                continue
            data = json.loads(m.group())
            return SuggestionResponse(
                over_by         = over_by,
                food_suggestion = data.get("food_suggestion", "Cucumber raita"),
                food_calories   = int(data.get("food_calories", 50)),
                food_note       = data.get("food_note", "Light and healthy."),
                exercise        = data.get("exercise", "Walk 2 km"),
                exercise_note   = data.get("exercise_note", "Burns ~100 kcal."),
                skipping_warning= data.get("skipping_warning",
                    "Skipping meals slows metabolism — eat light instead."),
            )
        except Exception:
            continue

    return _static_suggestion(over_by)


def _static_suggestion(over_by: float) -> SuggestionResponse:
    return SuggestionResponse(
        over_by         = over_by,
        food_suggestion = "Plain chaas (buttermilk)",
        food_calories   = 30,
        food_note       = "Only ~30 kcal, keeps you full and aids digestion.",
        exercise        = f"Walk {round(over_by / 60, 1)} km",
        exercise_note   = f"Burns approximately {min(round(over_by * 0.6), 300)} kcal.",
        skipping_warning= "Skipping meals is not recommended — eat light instead.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# OCR — MESS MENU SETUP (one-time)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/ocr/scan", response_model=OCRMenuResponse)
async def ocr_scan(file: UploadFile = File(...)):
    """
    Photo of mess menu board → matched dishes with nutrition + natural units.
    One-time setup. Frontend shows results for user to confirm before saving.
    Tries Google Vision API first, falls back to Tesseract.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(400, f"Invalid file type: {file.content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Empty file.")

    try:
        from ocr import extract_menu_dishes
        result = extract_menu_dishes(image_bytes)
        return OCRMenuResponse(**result)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")


@app.post("/ocr/save-menu")
async def save_menu(request: SaveMenuRequest):
    """
    Save confirmed mess menu dishes to PostgreSQL.
    Called after user reviews OCR results and confirms.
    Also upserts menu into ChromaDB for RAG context.
    """
    if not request.dishes:
        raise HTTPException(400, "No dishes provided.")

    from database import save_mess_menu, get_mess_menu_grouped
    from rag import upsert_menu

    count = await save_mess_menu(request.dishes, request.institution_id)

    # Upsert into ChromaDB so chatbot has menu context
    try:
        menu_grouped = await get_mess_menu_grouped(request.institution_id)
        upsert_menu(menu_grouped)
    except Exception as e:
        log.warning(f"RAG menu upsert failed (non-fatal): {e}")

    return {
        "status"        : "ok",
        "dishes_saved"  : count,
        "institution_id": request.institution_id,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ONBOARDING AGENT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/onboarding/start", response_model=OnboardingResponse)
def onboarding_start_endpoint(request: OnboardingStartRequest):
    """
    Start conversational onboarding session.
    Returns first question to show the user.
    """
    from agent import onboarding_start
    result = onboarding_start(request.session_id, request.user_id)
    return OnboardingResponse(**result)


@app.post("/onboarding/reply", response_model=OnboardingResponse)
async def onboarding_reply_endpoint(request: OnboardingReplyRequest):
    """
    Send user's reply → get next question or completed plan.
    When done=True, plan is saved to PostgreSQL automatically.

    Frontend flow:
        1. POST /onboarding/start → show question
        2. User replies → POST /onboarding/reply → show next question
        3. Repeat until done=True → profile + plan saved to DB
    """
    from agent import onboarding_reply
    result = await onboarding_reply(request.session_id, request.message)
    return OnboardingResponse(**{
        k: result.get(k)
        for k in OnboardingResponse.model_fields
    })


@app.get("/onboarding/state/{session_id}")
def onboarding_state(session_id: str):
    """Get current onboarding session state."""
    from agent import _onboarding_sessions
    state = _onboarding_sessions.get(session_id)
    if not state:
        raise HTTPException(404, "Onboarding session not found.")
    return {
        "session_id"    : session_id,
        "phase"         : state.get("phase"),
        "profile"       : state.get("profile", {}),
        "done"          : state.get("phase") == "done",
        "messages_count": len(state.get("messages", [])),
    }


@app.delete("/onboarding/{session_id}")
def onboarding_delete(session_id: str):
    from agent import onboarding_clear
    onboarding_clear(session_id)
    return {"message": f"Session {session_id} cleared."}


# ══════════════════════════════════════════════════════════════════════════════
# USER PROFILE
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/user/{user_id}")
async def get_user(user_id: str):
    """
    Fetch user profile + nutrition targets from PostgreSQL.
    Called by frontend on app load to check if onboarding is done.
    """
    from database import get_user_profile
    profile = await get_user_profile(user_id)
    if not profile:
        raise HTTPException(404, "User not found. Please complete onboarding.")
    return profile


@app.post("/user/{user_id}/update-weight")
async def update_weight(user_id: str, new_weight: float):
    """Update user's current weight — recalculates targets."""
    from database import get_user_profile, save_user_profile
    from agent import _compute_plan

    profile = await get_user_profile(user_id)
    if not profile:
        raise HTTPException(404, "User not found.")

    # Map DB columns back to profile dict format
    profile_dict = {
        "name"         : profile.get("name"),
        "age"          : profile.get("age"),
        "gender"       : profile.get("gender"),
        "height"       : profile.get("height_cm"),
        "weight"       : new_weight,
        "goal"         : profile.get("goal"),
        "diet"         : profile.get("diet"),
        "eats_in_mess" : profile.get("eats_in_mess"),
        "activities"   : profile.get("activities", []),
        "gym_days"     : profile.get("gym_days", 0),
        "gym_type"     : profile.get("gym_type"),
        "sleep"        : profile.get("sleep_hours", 7),
        "target_weight": profile.get("target_weight"),
        "duration"     : profile.get("duration_weeks"),
    }

    new_plan = _compute_plan(profile_dict)
    await save_user_profile(user_id, profile_dict, new_plan)

    return {"status": "ok", "new_weight": new_weight, "new_targets": new_plan}


# ══════════════════════════════════════════════════════════════════════════════
# MEAL LOGGING
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/log/meal")
async def log_meal_endpoint(request: LogMealRequest):
    """
    Log a meal item for a user.
    Source can be: menu (selected from mess menu), scan (photo), custom (typed).
    Portion multiplier handles natural serving adjustments.
    """
    from database import log_meal

    log_date = None
    if request.log_date:
        try:
            from datetime import datetime
            log_date = datetime.strptime(request.log_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(400, "log_date must be YYYY-MM-DD format.")

    dish = {
        "dish_key"          : request.dish_key,
        "display_name"      : request.display_name,
        "calories"          : request.calories,
        "protein"           : request.protein,
        "carbs"             : request.carbs,
        "fats"              : request.fats,
        "serving_desc"      : request.serving_desc,
        "portion_multiplier": request.portion_multiplier,
        "notes"             : request.notes,
    }

    entry_id = await log_meal(
        user_id  = request.user_id,
        meal_slot= request.meal_slot,
        dish     = dish,
        log_date = log_date,
        source   = request.source,
    )

    return {"status": "ok", "log_id": entry_id}


@app.delete("/log/{log_id}")
async def delete_log_entry(log_id: int, request: DeleteLogRequest):
    """Delete a specific logged meal entry."""
    from database import delete_meal_log
    deleted = await delete_meal_log(log_id, request.user_id)
    if not deleted:
        raise HTTPException(404, "Log entry not found or not yours.")
    return {"status": "ok", "deleted_id": log_id}


@app.get("/log/today/{user_id}")
async def get_today_logs(user_id: str):
    """
    Fetch all meal entries logged today.
    Used by daily log tab to show what's been eaten.
    """
    from database import get_daily_logs
    logs = await get_daily_logs(user_id)
    return {"user_id": user_id, "date": date.today().isoformat(), "logs": logs}


@app.get("/log/totals/{user_id}")
async def get_totals(user_id: str):
    """
    Today's macro totals + remaining vs targets.
    Used by progress bar in daily log tab.
    Shows: eaten kcal, protein, carbs, fats + how much left.
    """
    from database import get_today_totals
    totals = await get_today_totals(user_id)
    return totals


@app.get("/log/history/{user_id}")
async def get_log_history(
    user_id   : str,
    start_date: str,
    end_date  : str,
):
    """
    Fetch meal logs for a date range.
    Used by history tab.
    Dates in YYYY-MM-DD format.
    """
    from database import get_date_range_logs
    from datetime import datetime

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end   = datetime.strptime(end_date,   "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "Dates must be YYYY-MM-DD format.")

    if (end - start).days > 90:
        raise HTTPException(400, "Date range cannot exceed 90 days.")

    logs = await get_date_range_logs(user_id, start, end)
    return {"user_id": user_id, "start_date": start_date,
            "end_date": end_date, "logs": logs}


# ══════════════════════════════════════════════════════════════════════════════
# GAP ANALYSIS AGENT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/gap/start")
async def start_gap_analysis(request: GapAnalysisRequest):
    """
    Start gap analysis for a user's weekly menu selection.

    Frontend sends:
        user_id     : user identifier
        weekly_menu : { "monday": { "breakfast": [...], "lunch": [...] }, ... }

    Agent:
        1. Computes daily macro totals from selected dishes
        2. Finds days where calorie or protein gap > 10% of target
        3. Generates specific food recommendations to fill gaps
        4. Pauses for HITL — returns first recommendation for user to confirm

    Returns first pending recommendation immediately.
    """
    from database import get_user_profile
    from agent import run_gap_analysis

    profile = await get_user_profile(request.user_id)
    if not profile:
        raise HTTPException(404, "User not found. Complete onboarding first.")

    user_plan = {
        "calories": profile.get("calories", 2000),
        "protein" : profile.get("protein_g", 120),
        "carbs"   : profile.get("carbs_g", 250),
        "fats"    : profile.get("fats_g", 55),
    }

    result = await run_gap_analysis(
        user_id    = request.user_id,
        user_plan  = user_plan,
        weekly_menu= request.weekly_menu,
    )

    return result


@app.post("/gap/confirm")
async def confirm_gap_recommendation(request: GapConfirmRequest):
    """
    Human-in-the-loop response for gap analysis.

    Frontend shows the pending recommendation, user taps:
        "Add to plan" → accepted=True
        "Skip"        → accepted=False

    Returns next pending recommendation or status=done.
    """
    from agent import confirm_recommendation

    result = await confirm_recommendation(
        user_id  = request.user_id,
        day      = request.day,
        accepted = request.accepted,
    )

    return result


@app.get("/gap/status/{user_id}")
async def gap_analysis_status(user_id: str):
    """
    Get current gap analysis state for a user.
    Used by frontend to resume if user closes and reopens the app mid-flow.
    """
    from agent import get_gap_graph
    graph     = get_gap_graph()
    thread_id = f"gap_{user_id}"
    config    = {"configurable": {"thread_id": thread_id}}

    try:
        state = await graph.aget_state(config)
        if not state or not state.values:
            return {"status": "not_started"}
        return {
            "status"         : state.values.get("status", "unknown"),
            "pending_confirm": state.values.get("pending_confirm"),
            "gaps_count"     : len(state.values.get("gaps", [])),
            "confirmed_count": len(state.values.get("confirmed", [])),
        }
    except Exception:
        return {"status": "not_started"}


# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY REVIEW
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/review/{user_id}", response_model=WeeklyReviewResponse)
async def weekly_review(user_id: str):
    """
    Generate 7-day review for a user.
    Reads from PostgreSQL — no localStorage needed.

    Returns:
        stats    : per-day breakdown + 7-day averages
        insights : specific observations (calorie gap, protein, consistency)
        summary  : LLM narrative or rule-based fallback
    """
    from database import get_user_profile
    from agent import run_weekly_review

    profile = await get_user_profile(user_id)
    if not profile:
        raise HTTPException(404, "User not found.")

    plan = {
        "goal"    : profile.get("goal", "maintain"),
        "calories": profile.get("calories", 2000),
        "protein" : profile.get("protein_g", 120),
    }

    try:
        result = await run_weekly_review(user_id, plan)
        return WeeklyReviewResponse(**result)
    except Exception as e:
        log.error(f"Weekly review failed: {e}")
        raise HTTPException(500, f"Weekly review failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Send a message to NutriAI chatbot → full response.
    Context-aware: knows user's profile, today's intake, meal history.
    Persistent: conversation continues across sessions via PostgreSQL.
    """
    from database import get_user_profile
    from chatbot import chat

    profile = await get_user_profile(request.user_id)
    if not profile:
        raise HTTPException(404, "User not found. Complete onboarding first.")

    try:
        response = await chat(
            user_id     = request.user_id,
            message     = request.message,
            user_profile= profile,
            thread_id   = request.thread_id,
        )
        return {"response": response, "user_id": request.user_id}
    except Exception as e:
        log.error(f"Chat failed: {e}")
        raise HTTPException(500, f"Chat failed: {e}")


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Send a message → streaming response (token by token).
    Use this for the chat UI — shows text appearing in real time.
    Also streams tool call notifications:
        "[Looking up nutrition info...]"
        "[Checking today's intake...]"

    Frontend usage:
        const resp = await fetch('/chat/stream', { method: 'POST', body: ... })
        const reader = resp.body.getReader()
        // read chunks as they arrive
    """
    from database import get_user_profile
    from chatbot import stream_chat

    profile = await get_user_profile(request.user_id)
    if not profile:
        raise HTTPException(404, "User not found. Complete onboarding first.")

    async def generate():
        try:
            async for token in stream_chat(
                user_id     = request.user_id,
                message     = request.message,
                user_profile= profile,
                thread_id   = request.thread_id,
            ):
                yield token
        except Exception as e:
            log.error(f"Stream chat failed: {e}")
            yield f"\n[Error: {e}]"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 20):
    """
    Fetch recent conversation history.
    Used by frontend to restore chat on app open.
    Returns list of { role, content } dicts.
    """
    from chatbot import get_chat_history
    history = await get_chat_history(user_id, limit=limit)
    return {"user_id": user_id, "history": history, "count": len(history)}


@app.delete("/chat/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear conversation history — start fresh."""
    from chatbot import clear_chat_history
    await clear_chat_history(user_id)
    return {"status": "ok", "message": "Chat history cleared."}


# ══════════════════════════════════════════════════════════════════════════════
# RAG — DIRECT NUTRITION Q&A
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/rag/ask", response_model=RAGResponse)
async def rag_ask(request: RAGRequest):
    """
    Direct RAG nutrition Q&A — bypasses chatbot conversation flow.
    Used for one-off nutrition questions without chat context.
    """
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    try:
        from rag import answer_question
        result = answer_question(
            question   = request.question,
            user_goal  = request.user_goal,
            user_id    = request.user_id,
            today_menu = request.today_menu,
            user_log   = request.user_log,
        )
        return RAGResponse(**result)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"RAG failed: {e}")


@app.post("/rag/populate")
def rag_populate(force: bool = False):
    """
    Re-populate ChromaDB with nutrition_db.
    Admin endpoint — call if you update nutrition_db.py.
    """
    try:
        from rag import populate_nutrition_db
        count = populate_nutrition_db(force=force)
        return {"status": "ok", "docs_in_db": count}
    except RuntimeError as e:
        raise HTTPException(503, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# MESS MENU — READ
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/menu/{institution_id}")
async def get_menu(institution_id: str = "default"):
    """
    Fetch the full mess menu grouped by meal slot.
    Used by frontend dropdowns when user selects weekly meals.
    Returns: { breakfast: [...], lunch: [...], snacks: [...], dinner: [...] }
    """
    from database import get_mess_menu_grouped
    menu = await get_mess_menu_grouped(institution_id)
    total = sum(len(v) for v in menu.values())
    if total == 0:
        raise HTTPException(404, "No menu found. Please scan and save the mess menu first.")
    return {"institution_id": institution_id, "menu": menu, "total_dishes": total}
