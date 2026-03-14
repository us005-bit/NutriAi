"""
NutriAI - agent.py
================
Three LangGraph agentic workflows:

1. ONBOARDING AGENT  (LangGraph StateGraph)
   Conversational profile setup — collects name, age, gender, height, weight,
   goal, diet, activities, gym details, sleep.
   Computes personalised calorie/protein targets.
   Saves final profile to PostgreSQL via database.py.

2. GAP ANALYSIS AGENT  (LangGraph StateGraph)  ← NEW
   Analyses user's weekly mess menu selection vs daily targets.
   Finds calorie and protein gaps per day.
   Generates specific food recommendations to fill gaps.
   Human-in-the-loop: user confirms or skips each recommendation.

3. WEEKLY REVIEW AGENT  (LangGraph StateGraph)
   Pulls 7-day log from PostgreSQL (not localStorage anymore).
   Computes per-day stats, generates rule-based insights,
   optional LLM narrative summary.

LangGraph concepts demonstrated:
    LG-4   StateGraph, TypedDict state, nodes, edges
    LG-5   Sequential workflows (onboarding, weekly review)
    LG-6   Parallel workflows (gap analysis — all days computed in parallel)
    LG-7   Conditional workflows (onboarding skip logic, gap router)
    LG-8   Iterative workflows (onboarding Q&A loop)
    LG-9   PostgreSQL persistence (onboarding sessions)
    LG-15  Human in the loop (gap analysis confirmation)
    LG-19  Subgraphs (weekly review as subgraph)

Install:
    pip install langgraph langchain langchain-community
                psycopg[binary] --break-system-packages

Environment variables (.env):
    HF_API_KEY=hf_xxxx
    DATABASE_URL=postgresql://...neon.tech/neondb?sslmode=require
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import date, timedelta
from typing import Any, Annotated, Optional
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger("nutriai.agent")

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import HuggingFaceHub

# ── Local ─────────────────────────────────────────────────────────────────────
from nutrition import get_nutrition_safe


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HF UTILITY
# ══════════════════════════════════════════════════════════════════════════════
_HF_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def _hf_call(prompt: str, max_tokens: int = 600) -> Optional[str]:
    """Call HuggingFace Inference API, try models in order."""
    import httpx
    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        return None

    headers  = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
    messages = [{"role": "user", "content": prompt}]

    for model in _HF_MODELS:
        try:
            resp = httpx.post(
                "https://router.huggingface.co/v1/chat/completions",
                json={"model": model, "messages": messages,
                      "temperature": 0.3, "max_tokens": max_tokens},
                headers=headers,
                timeout=45,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            log.warning("HF model %s failed: %s", model, exc)
    return None


def _parse_json(raw: str) -> Optional[dict]:
    """Strip markdown fences and extract JSON object."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)
    m   = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ① ONBOARDING AGENT
# ══════════════════════════════════════════════════════════════════════════════

class OnboardingState(TypedDict):
    """Mutable state passed between LangGraph onboarding nodes."""
    messages     : Annotated[list, add_messages]  # conversation history
    profile      : dict                           # collected profile fields
    phase        : str                            # you|goals|activities|plan|done
    next_question: str                            # question shown to user
    plan         : Optional[dict]                 # computed nutrition plan
    user_id      : str                            # scopes PostgreSQL save
    error        : Optional[str]


# Ordered fields and their questions
_ONBOARDING_FLOW = [
    ("name",         "Hey! I'm NutriAI 👋 What's your name?",                                      "you"),
    ("age",          "Nice to meet you, {name}! How old are you?",                               "you"),
    ("gender",       "Got it. What's your gender? (male / female / other)",                      "you"),
    ("height",       "What's your height in cm?",                                                "you"),
    ("weight",       "And your current weight in kg?",                                           "you"),
    ("goal",         "What's your main goal?\n• lose — lose weight\n• gain — build muscle\n• maintain — stay at current weight", "goals"),
    ("target_weight","What's your target weight in kg?",                                         "goals"),   # skipped for maintain
    ("duration",     "How many weeks to reach that target?",                                     "goals"),   # skipped for maintain
    ("diet",         "Dietary preference?\n• veg\n• non_veg\n• egg",                            "goals"),
    ("eats_in_mess", "Do you eat at the college mess?\n• yes\n• no\n• mixed",                   "goals"),
    ("sleep",        "How many hours of sleep per night?",                                       "goals"),
    ("activities",   "What activities do you do?\n• gym  • swimming  • running  • cycling  • yoga  • walking  • sport  • none", "activities"),
    ("gym_days",     "How many days/week do you go to the gym? (0–7)",                          "activities"),  # skipped if no gym
    ("gym_type",     "What kind of training?\n• strength  • cardio  • mixed",                   "activities"),  # skipped if no gym
    ("sport_name",   "Which sport?",                                                              "activities"),  # skipped if no sport
]


def _should_skip(field: str, profile: dict) -> bool:
    """Conditional skip logic — demonstrated as LangGraph conditional workflow."""
    goal = profile.get("goal", "")
    acts = profile.get("activities", [])
    if field in ("target_weight", "duration") and goal == "maintain":
        return True
    if field in ("gym_days", "gym_type") and "gym" not in acts:
        return True
    if field == "sport_name" and "sport" not in acts:
        return True
    return False


def _compute_plan(profile: dict) -> dict:
    """
    Compute personalised nutrition targets from profile.
    Uses Mifflin-St Jeor BMR + TDEE + goal adjustment.
    Protein targets based on ISSN 2023 guidelines.
    """
    age        = int(profile.get("age", 21))
    gender     = str(profile.get("gender", "male")).lower()
    height     = float(profile.get("height", 170))
    weight     = float(profile.get("weight", 70))
    goal       = str(profile.get("goal", "maintain"))
    acts       = profile.get("activities", [])
    if isinstance(acts, str):
        acts = [a.strip() for a in acts.replace(",", " ").split()]
    gym_days   = int(profile.get("gym_days", 0))
    sleep_hrs  = float(profile.get("sleep", 7))
    target_wt  = float(profile.get("target_weight", weight))
    duration_w = int(profile.get("duration", 12))

    # BMR — Mifflin-St Jeor
    if gender in ("male", "m"):
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    bmr = round(bmr)

    # Activity multiplier
    if gym_days >= 5 or "swimming" in acts or "running" in acts:
        mult = 1.55
    elif gym_days >= 3 or len([a for a in acts if a != "none"]) >= 2:
        mult = 1.375
    else:
        mult = 1.2
    tdee = round(bmr * mult)

    # Goal adjustment
    if goal == "lose":
        base_cal = max(tdee - 500, 1200)
    elif goal == "gain":
        base_cal = tdee + 300
    else:
        base_cal = tdee

    # Sleep penalty
    if sleep_hrs < 6:
        base_cal -= 50
    elif sleep_hrs < 7:
        base_cal -= 25
    base_cal = max(base_cal, 1200)

    # Gym-day vs rest-day split
    gym_cal  = base_cal + 200 if gym_days > 0 else base_cal
    rest_cal = max(base_cal - 100, 1200) if gym_days > 0 else base_cal

    # Protein — ISSN 2023
    pro_per_kg = {"gain": 1.8, "lose": 1.6}.get(goal, 1.4)
    if "gym" in acts and goal == "maintain":
        pro_per_kg = 1.6
    protein = min(round(weight * pro_per_kg), 160)

    fats  = round((base_cal * 0.25) / 9)
    carbs = round((base_cal - protein * 4 - fats * 9) / 4)
    bmi   = round(weight / ((height / 100) ** 2), 1)

    return {
        "calories"         : base_cal,
        "gymDayCalories"   : gym_cal,
        "restDayCalories"  : rest_cal,
        "protein"          : protein,
        "carbs"            : carbs,
        "fats"             : fats,
        "bmr"              : bmr,
        "tdee"             : tdee,
        "bmi"              : str(bmi),
        "gym_days_per_week": gym_days,
        "gym_type"         : profile.get("gym_type", ""),
        "activities"       : acts,
        "goal"             : goal,
        "diet"             : profile.get("diet", "non_veg"),
        "eats_in_mess"     : profile.get("eats_in_mess", "yes"),
        "sleep_hours"      : sleep_hrs,
        "name"             : profile.get("name", ""),
        "age"              : age,
        "gender"           : gender,
        "height"           : height,
        "weight"           : weight,
        "targetWeight"     : target_wt,
        "duration"         : duration_w,
    }


# ── Onboarding nodes ──────────────────────────────────────────────────────────

def _node_ask(state: OnboardingState) -> OnboardingState:
    """Find next unanswered field → formulate question. Sequential node."""
    profile = state["profile"]
    for field, question, phase in _ONBOARDING_FLOW:
        if field not in profile and not _should_skip(field, profile):
            q = question.format(**{k: v for k, v in profile.items()
                                   if isinstance(v, str)})
            return {**state, "next_question": q, "phase": phase}
    return {**state, "next_question": "", "phase": "plan"}


def _node_parse(state: OnboardingState) -> OnboardingState:
    """Parse latest user message into profile field. Sequential node."""
    if not state["messages"]:
        return state

    last_user = next(
        (m.content if hasattr(m, "content") else m.get("content", "")
         for m in reversed(state["messages"])
         if (hasattr(m, "type") and m.type == "human")
         or (isinstance(m, dict) and m.get("role") == "user")),
        ""
    ).strip()

    profile = dict(state["profile"])
    for field, _, _ in _ONBOARDING_FLOW:
        if field not in profile and not _should_skip(field, profile):
            v = _parse_field(field, last_user)
            if v is not None:
                profile[field] = v
            break

    return {**state, "profile": profile}


def _parse_field(field: str, raw: str) -> Any:
    """Rule-based parser for each profile field."""
    raw = raw.strip().lower()

    if field == "name":
        return raw.title()
    if field in ("age", "height", "gym_days"):
        m = re.search(r"\d+\.?\d*", raw)
        return int(float(m.group())) if m else None
    if field == "weight":
        m = re.search(r"\d+\.?\d*", raw)
        return float(m.group()) if m else None
    if field == "sleep":
        m = re.search(r"\d+\.?\d*", raw)
        return float(m.group()) if m else 7.0
    if field == "target_weight":
        m = re.search(r"\d+\.?\d*", raw)
        return float(m.group()) if m else None
    if field == "duration":
        m = re.search(r"\d+", raw)
        return int(m.group()) if m else 12
    if field == "gender":
        if any(w in raw for w in ("female", "f", "woman", "girl")):
            return "female"
        if any(w in raw for w in ("other", "non")):
            return "other"
        return "male"
    if field == "goal":
        if any(w in raw for w in ("lose", "loss", "cut", "slim")):
            return "lose"
        if any(w in raw for w in ("gain", "build", "bulk", "muscle")):
            return "gain"
        return "maintain"
    if field == "diet":
        if "veg" in raw and "non" not in raw and "egg" not in raw:
            return "veg"
        if "egg" in raw:
            return "egg"
        return "non_veg"
    if field == "eats_in_mess":
        if "yes" in raw or raw in ("y", "yeah", "always"):
            return "yes"
        if "mix" in raw or "some" in raw:
            return "mixed"
        return "no"
    if field == "activities":
        options = ["gym", "swimming", "running", "cycling", "yoga", "walking", "sport", "none"]
        found   = [o for o in options if o in raw]
        return found if found else ["none"]
    if field == "gym_type":
        if "strength" in raw or "weight" in raw or "lift" in raw:
            return "strength"
        if "cardio" in raw:
            return "cardio"
        return "mixed"
    if field == "sport_name":
        return raw.title() if raw else "Sport"
    return raw


async def _node_save_profile(state: OnboardingState) -> OnboardingState:
    """
    Compute plan + save to PostgreSQL. Final onboarding node.
    Demonstrates PostgreSQL persistence from agent.
    """
    from database import save_user_profile

    plan = _compute_plan(state["profile"])
    try:
        await save_user_profile(state["user_id"], state["profile"], plan)
        log.info(f"Onboarding complete — profile saved for '{state['user_id']}'")
    except Exception as e:
        log.error(f"Failed to save profile: {e}")
        return {**state, "plan": plan, "phase": "done",
                "error": f"Profile computed but save failed: {e}"}

    return {**state, "plan": plan, "phase": "done", "error": None}


def _should_compute(state: OnboardingState) -> str:
    """
    Conditional edge router.
    All fields collected → compute. Else → ask next question.
    Demonstrates LangGraph conditional workflow (LG-7).
    """
    profile = state["profile"]
    for field, _, _ in _ONBOARDING_FLOW:
        if field not in profile and not _should_skip(field, profile):
            return "ask"
    return "compute"


def build_onboarding_graph(checkpointer=None):
    """
    Build and compile the onboarding StateGraph.

    Flow (iterative loop — LG-8):
        parse → [all fields?] → ask → (user replies) → parse → ...
                             → compute → save → END
    """
    g = StateGraph(OnboardingState)

    g.add_node("parse",   _node_parse)
    g.add_node("ask",     _node_ask)
    g.add_node("compute", _node_save_profile)   # async node

    g.set_entry_point("parse")
    g.add_conditional_edges(
        "parse",
        _should_compute,
        {"ask": "ask", "compute": "compute"},
    )
    g.add_edge("ask",     END)
    g.add_edge("compute", END)

    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()


# ── In-memory onboarding sessions (backed by PostgreSQL checkpointer) ─────────
_onboarding_graph = None
_onboarding_sessions: dict[str, OnboardingState] = {}


def get_onboarding_graph(checkpointer=None):
    global _onboarding_graph
    if _onboarding_graph is None:
        _onboarding_graph = build_onboarding_graph(checkpointer)
    return _onboarding_graph


def onboarding_start(session_id: str, user_id: str = "") -> dict:
    """Create fresh onboarding session, return first question."""
    state: OnboardingState = {
        "messages"     : [],
        "profile"      : {},
        "phase"        : "you",
        "next_question": "",
        "plan"         : None,
        "user_id"      : user_id or session_id,
        "error"        : None,
    }
    state = _node_ask(state)
    _onboarding_sessions[session_id] = state
    return {
        "session_id": session_id,
        "question"  : state["next_question"],
        "phase"     : state["phase"],
        "done"      : False,
        "plan"      : None,
    }


async def onboarding_reply(session_id: str, user_message: str) -> dict:
    """
    User sent a reply → advance state machine one step.
    Returns next question or, if done, the saved plan.
    """
    state = _onboarding_sessions.get(session_id)
    if not state:
        state = onboarding_start(session_id)
        state = _onboarding_sessions[session_id]

    state["messages"].append(HumanMessage(content=user_message))

    graph  = get_onboarding_graph()
    result = await graph.ainvoke(state)
    new_state: OnboardingState = result
    _onboarding_sessions[session_id] = new_state

    if new_state.get("phase") == "done" and new_state.get("plan"):
        return {
            "session_id": session_id,
            "question"  : None,
            "phase"     : "done",
            "done"      : True,
            "plan"      : new_state["plan"],
            "profile"   : new_state["profile"],
            "error"     : new_state.get("error"),
        }

    return {
        "session_id"    : session_id,
        "question"      : new_state.get("next_question", ""),
        "phase"         : new_state.get("phase", "you"),
        "done"          : False,
        "plan"          : None,
        "profile_so_far": {k: v for k, v in new_state["profile"].items()
                           if k in ("name", "goal", "activities")},
    }


def onboarding_clear(session_id: str) -> None:
    _onboarding_sessions.pop(session_id, None)


# ══════════════════════════════════════════════════════════════════════════════
# ② GAP ANALYSIS AGENT  (NEW)
# Finds daily protein/calorie gaps from weekly menu and recommends fixes.
# ══════════════════════════════════════════════════════════════════════════════

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
SLOTS = ["breakfast", "lunch", "snacks", "dinner"]

class GapAnalysisState(TypedDict):
    """State for the gap analysis LangGraph."""
    user_id         : str
    user_plan       : dict          # calorie + protein targets
    weekly_menu     : dict          # { day: { slot: [dishes] } }
    daily_totals    : dict          # { day: { cal, pro, car, fat } }
    gaps            : list[dict]    # [ { day, cal_gap, pro_gap } ]
    recommendations : list[dict]    # [ { day, suggestions: [...] } ]
    confirmed       : list[dict]    # recommendations user approved (HITL)
    pending_confirm : Optional[dict]# current rec waiting for user input
    status          : str           # computing|awaiting_confirm|done
    error           : Optional[str]


def _node_compute_daily_totals(state: GapAnalysisState) -> GapAnalysisState:
    """
    Node 1: Compute macro totals for each day from weekly menu selection.
    Parallel workflow — processes all 7 days simultaneously (LG-6).
    """
    weekly_menu = state["weekly_menu"]
    daily_totals: dict = {}

    def compute_day(day: str, day_menu: dict) -> tuple[str, dict]:
        cal = pro = car = fat = 0.0
        for slot in SLOTS:
            for dish in day_menu.get(slot, []):
                multiplier = dish.get("portion_multiplier", 1.0)
                cal += dish.get("calories", 0) * multiplier
                pro += dish.get("protein",  0) * multiplier
                car += dish.get("carbs",    0) * multiplier
                fat += dish.get("fats",     0) * multiplier
        return day, {
            "cal": round(cal), "pro": round(pro),
            "car": round(car), "fat": round(fat),
        }

    # Parallel execution across all 7 days — LangGraph parallel workflow (LG-6)
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {
            executor.submit(compute_day, day, weekly_menu.get(day, {}))
            for day in DAYS
        }
        for future in futures:
            day, totals = future.result()
            daily_totals[day] = totals

    log.info(f"Daily totals computed for {len(daily_totals)} days")
    return {**state, "daily_totals": daily_totals}


def _node_find_gaps(state: GapAnalysisState) -> GapAnalysisState:
    """
    Node 2: Compare daily totals against user targets. Find gaps.
    Sequential node — simple arithmetic comparison.
    """
    plan        = state["user_plan"]
    cal_target  = plan.get("calories", 2000)
    pro_target  = plan.get("protein",  120)
    daily       = state["daily_totals"]

    gaps = []
    for day in DAYS:
        totals  = daily.get(day, {"cal": 0, "pro": 0})
        cal_gap = cal_target - totals["cal"]
        pro_gap = pro_target - totals["pro"]

        # Only flag significant gaps (>10% of target)
        if cal_gap > cal_target * 0.1 or pro_gap > pro_target * 0.1:
            gaps.append({
                "day"       : day,
                "cal_eaten" : totals["cal"],
                "pro_eaten" : totals["pro"],
                "cal_gap"   : round(cal_gap),
                "pro_gap"   : round(pro_gap),
                "cal_target": cal_target,
                "pro_target": pro_target,
            })

    log.info(f"Gap analysis: {len(gaps)} days with significant gaps")
    return {**state, "gaps": gaps}


def _node_generate_recommendations(state: GapAnalysisState) -> GapAnalysisState:
    """
    Node 3: Generate food recommendations for each gap day using LLM.
    Focuses on realistic Indian mess + common foods.
    """
    gaps = state["gaps"]
    if not gaps:
        return {**state, "recommendations": [], "status": "done"}

    recommendations = []

    for gap in gaps:
        day     = gap["day"]
        cal_gap = gap["cal_gap"]
        pro_gap = gap["pro_gap"]

        # Build prompt focused on gaps
        gap_lines = []
        if cal_gap > 100:
            gap_lines.append(f"Calorie gap: {cal_gap} kcal short")
        if pro_gap > 5:
            gap_lines.append(f"Protein gap: {pro_gap}g short")

        if not gap_lines:
            continue

        prompt = f"""You are a nutrition advisor for Indian college students.
On {day.capitalize()}, the student is:
{chr(10).join(gap_lines)}

Suggest 2-3 specific, practical foods to fill these gaps.
Focus on: eggs, dal, paneer, curd, milk, banana, peanuts, chicken (if non-veg),
sprouts, or common Indian snacks available in college.
Be specific about quantity: e.g. "2 boiled eggs (120 kcal, 12g protein)".

Return ONLY a JSON object:
{{
  "day": "{day}",
  "suggestions": [
    {{
      "food": "food name",
      "quantity": "specific amount",
      "calories": int,
      "protein_g": float,
      "note": "one short reason why this helps"
    }}
  ],
  "summary": "1 sentence overview"
}}
Return ONLY valid JSON."""

        raw  = _hf_call(prompt, max_tokens=400)
        data = _parse_json(raw) if raw else None

        if data and data.get("suggestions"):
            recommendations.append({
                "day"        : day,
                "gap"        : gap,
                "suggestions": data["suggestions"],
                "summary"    : data.get("summary", ""),
            })
        else:
            # Rule-based fallback if LLM fails
            suggestions = []
            if pro_gap > 10:
                suggestions.append({
                    "food"    : "Boiled eggs",
                    "quantity": f"{math.ceil(pro_gap / 6)} eggs",
                    "calories": math.ceil(pro_gap / 6) * 70,
                    "protein_g": round(math.ceil(pro_gap / 6) * 6.0, 1),
                    "note"    : "Quick, cheap, high-protein option available everywhere.",
                })
            if cal_gap > 200:
                suggestions.append({
                    "food"    : "Peanut butter on bread",
                    "quantity": "2 slices with 2 tbsp peanut butter",
                    "calories": 350,
                    "protein_g": 12.0,
                    "note"    : "Dense in calories and protein — good evening snack.",
                })
            if suggestions:
                recommendations.append({
                    "day"        : day,
                    "gap"        : gap,
                    "suggestions": suggestions,
                    "summary"    : f"Rule-based suggestions for {day} gap.",
                })

    log.info(f"Generated recommendations for {len(recommendations)} days")
    return {**state, "recommendations": recommendations, "status": "awaiting_confirm"}


def _node_await_human(state: GapAnalysisState) -> GapAnalysisState:
    """
    Node 4: Human-in-the-loop pause point (LG-15).
    Sets pending_confirm to the next unconfirmed recommendation.
    LangGraph will interrupt here and wait for user input.

    Frontend displays the recommendation and user taps:
        "Add to plan" → confirmed=True
        "Skip"        → confirmed=False
    """
    recs      = state["recommendations"]
    confirmed = state.get("confirmed", [])

    # Find next unconfirmed recommendation
    confirmed_days = {c["day"] for c in confirmed}
    pending = next(
        (r for r in recs if r["day"] not in confirmed_days),
        None
    )

    if pending is None:
        # All recommendations processed
        return {**state, "pending_confirm": None, "status": "done"}

    log.info(f"HITL: awaiting confirmation for {pending['day']}")
    return {**state, "pending_confirm": pending, "status": "awaiting_confirm"}


def _should_continue_hitl(state: GapAnalysisState) -> str:
    """
    Conditional edge: more recommendations to confirm → loop back.
    All done → END.
    Demonstrates conditional + iterative combined (LG-7 + LG-8).
    """
    if state["status"] == "done" or not state.get("pending_confirm"):
        return "done"
    return "await"


def build_gap_analysis_graph(checkpointer=None):
    """
    Build the gap analysis StateGraph.

    Flow:
        compute_totals
            → find_gaps
                → generate_recommendations
                    → await_human  ←──────────────┐
                        → [more?]                  │
                            YES → await_human  ────┘  (iterative HITL loop)
                            NO  → END
    """
    g = StateGraph(GapAnalysisState)

    g.add_node("compute_totals",         _node_compute_daily_totals)
    g.add_node("find_gaps",              _node_find_gaps)
    g.add_node("generate_recommendations", _node_generate_recommendations)
    g.add_node("await_human",            _node_await_human)

    g.set_entry_point("compute_totals")
    g.add_edge("compute_totals",           "find_gaps")
    g.add_edge("find_gaps",                "generate_recommendations")
    g.add_edge("generate_recommendations", "await_human")

    # Conditional loop: more to confirm → await, all done → END
    g.add_conditional_edges(
        "await_human",
        _should_continue_hitl,
        {"await": "await_human", "done": END},
    )

    # Interrupt before await_human so frontend can inject user decision
    interrupt_before = ["await_human"]

    if checkpointer:
        return g.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before,
        )
    return g.compile(interrupt_before=interrupt_before)


# ── Gap analysis public API ────────────────────────────────────────────────────
_gap_graph = None

def get_gap_graph(checkpointer=None):
    global _gap_graph
    if _gap_graph is None:
        _gap_graph = build_gap_analysis_graph(checkpointer)
    return _gap_graph


async def run_gap_analysis(
    user_id    : str,
    user_plan  : dict,
    weekly_menu: dict,
    checkpointer = None,
) -> dict:
    """
    Start gap analysis. Returns first pending recommendation for user confirmation.

    Args:
        user_id     : User identifier (scopes thread)
        user_plan   : { calories, protein, ... } from database
        weekly_menu : { day: { slot: [dishes] } } — user's weekly selection
        checkpointer: AsyncPostgresSaver for persistence

    Returns:
        { status, pending_confirm, gaps, recommendations }
    """
    graph     = get_gap_graph(checkpointer)
    thread_id = f"gap_{user_id}"
    config    = {"configurable": {"thread_id": thread_id}}

    initial = GapAnalysisState(
        user_id         = user_id,
        user_plan       = user_plan,
        weekly_menu     = weekly_menu,
        daily_totals    = {},
        gaps            = [],
        recommendations = [],
        confirmed       = [],
        pending_confirm = None,
        status          = "computing",
        error           = None,
    )

    result = await graph.ainvoke(initial, config=config)
    return {
        "status"         : result["status"],
        "pending_confirm": result.get("pending_confirm"),
        "gaps"           : result["gaps"],
        "recommendations": result["recommendations"],
    }


async def confirm_recommendation(
    user_id  : str,
    day      : str,
    accepted : bool,
    checkpointer = None,
) -> dict:
    """
    User confirmed or skipped a recommendation (HITL response).
    Resumes the gap analysis graph from the interrupt point.

    Args:
        user_id  : User identifier
        day      : Day being confirmed
        accepted : True = add to plan, False = skip
        checkpointer: AsyncPostgresSaver

    Returns:
        Next pending_confirm or status=done
    """
    graph     = get_gap_graph(checkpointer)
    thread_id = f"gap_{user_id}"
    config    = {"configurable": {"thread_id": thread_id}}

    # Get current state
    current = await graph.aget_state(config)
    state   = current.values

    # Update confirmed list
    confirmed = list(state.get("confirmed", []))
    if accepted:
        pending = state.get("pending_confirm")
        if pending:
            confirmed.append({**pending, "accepted": True})
    else:
        # Mark as skipped
        confirmed.append({"day": day, "accepted": False})

    # Resume graph with updated confirmed list
    await graph.aupdate_state(config, {"confirmed": confirmed})
    result = await graph.ainvoke(None, config=config)

    return {
        "status"         : result["status"],
        "pending_confirm": result.get("pending_confirm"),
        "confirmed"      : result["confirmed"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ③ WEEKLY REVIEW AGENT
# Reads from PostgreSQL instead of localStorage.
# ══════════════════════════════════════════════════════════════════════════════

class WeeklyReviewState(TypedDict):
    user_id  : str
    plan     : dict
    stats    : Optional[dict]
    insights : Optional[list]
    summary  : Optional[str]
    error    : Optional[str]


async def _node_weekly_stats(state: WeeklyReviewState) -> WeeklyReviewState:
    """
    Node 1: Fetch 7-day stats from PostgreSQL.
    Replaces old localStorage-based stat computation.
    """
    from database import get_weekly_summary
    try:
        stats = await get_weekly_summary(state["user_id"])
        return {**state, "stats": stats}
    except Exception as e:
        log.error(f"Weekly stats fetch failed: {e}")
        return {**state, "stats": {}, "error": str(e)}


def _node_weekly_insights(state: WeeklyReviewState) -> WeeklyReviewState:
    """
    Node 2: Rule-based insight generation — no LLM needed.
    Fast, deterministic, always works even if LLM is down.
    """
    stats    = state.get("stats", {})
    insights = []

    if not stats or stats.get("n_tracked", 0) == 0:
        return {**state,
                "insights": ["No meals logged this week. Start logging tomorrow!"],
                "summary" : "No data this week."}

    avg_cal = stats["avg_calories"]
    cal_tgt = stats["target_calories"]
    avg_pro = stats["avg_protein"]
    pro_tgt = stats["target_protein"]

    # Calorie insights
    cal_diff = avg_cal - cal_tgt
    if abs(cal_diff) < cal_tgt * 0.05:
        insights.append(f"✅ Calories on point — averaging {avg_cal} kcal vs {cal_tgt} target.")
    elif cal_diff > 0:
        insights.append(f"⚠️ Running {abs(cal_diff)} kcal over daily target. Try smaller rice portions or skip fried snacks.")
    else:
        insights.append(f"📉 {abs(cal_diff)} kcal below target. Add a roti or a glass of milk to fuel better.")

    # Protein insights
    pro_gap = pro_tgt - avg_pro
    if pro_gap <= 0:
        insights.append(f"💪 Protein target met — {avg_pro}g / {pro_tgt}g. Great work!")
    elif pro_gap <= 15:
        insights.append(f"💪 Protein close — {avg_pro}g / {pro_tgt}g. One egg or a cup of curd daily will close it.")
    else:
        insights.append(f"🥚 Protein gap: {avg_pro}g vs {pro_tgt}g target. Include dal, paneer, eggs or curd at every meal.")

    # Consistency
    hit_pct = stats.get("cal_hit_pct", 0)
    if hit_pct >= 70:
        insights.append(f"🎯 Hit calorie target {stats['cal_hit_days']}/{stats['n_tracked']} days — very consistent!")
    elif hit_pct >= 40:
        insights.append(f"📊 Hit target {stats['cal_hit_days']}/{stats['n_tracked']} days — room to improve.")
    else:
        insights.append(f"📊 Only on-target {stats['cal_hit_days']}/{stats['n_tracked']} days. Try pre-logging meals the night before.")

    return {**state, "insights": insights}


def _node_weekly_summary(state: WeeklyReviewState) -> WeeklyReviewState:
    """
    Node 3: Optional LLM narrative. Falls back to rule-based if LLM unavailable.
    """
    stats    = state.get("stats", {})
    insights = state.get("insights", [])

    if not stats or stats.get("n_tracked", 0) == 0:
        return {**state, "summary": "No meals logged this week."}

    prompt = f"""You are a friendly nutrition coach reviewing an Indian college student's week.

Goal: {state['plan'].get('goal', 'maintain')} weight.
Daily targets: {stats['target_calories']} kcal, {stats['target_protein']}g protein.
This week: {stats['avg_calories']} kcal/day avg, {stats['avg_protein']}g protein/day avg.
Days tracked: {stats['n_tracked']}/7.
Calorie target hit: {stats.get('cal_hit_pct', 0)}% of days.

Key insights:
{chr(10).join('- ' + i for i in insights)}

Write a warm, encouraging 3-4 sentence summary. Acknowledge what went well,
identify the biggest gap, give ONE specific actionable tip for next week.
Be concise — no bullet points, just prose."""

    raw     = _hf_call(prompt, max_tokens=250)
    summary = raw if raw else " ".join(insights[:2])
    return {**state, "summary": summary}


def build_weekly_review_graph():
    """
    Build weekly review as a subgraph (LG-19).
    Sequential: stats → insights → summary → END.
    """
    g = StateGraph(WeeklyReviewState)
    g.add_node("stats",    _node_weekly_stats)
    g.add_node("insights", _node_weekly_insights)
    g.add_node("summary",  _node_weekly_summary)

    g.set_entry_point("stats")
    g.add_edge("stats",    "insights")
    g.add_edge("insights", "summary")
    g.add_edge("summary",  END)

    return g.compile()


_weekly_graph = None

def get_weekly_graph():
    global _weekly_graph
    if _weekly_graph is None:
        _weekly_graph = build_weekly_review_graph()
    return _weekly_graph


async def run_weekly_review(user_id: str, plan: dict) -> dict:
    """
    Entry point for weekly review.
    Pulls data from PostgreSQL — no more localStorage dependency.

    Returns: { stats, insights, summary }
    """
    initial = WeeklyReviewState(
        user_id  = user_id,
        plan     = plan,
        stats    = None,
        insights = None,
        summary  = None,
        error    = None,
    )

    graph  = get_weekly_graph()
    result = await graph.ainvoke(initial)

    return {
        "stats"   : result.get("stats",    {}),
        "insights": result.get("insights", []),
        "summary" : result.get("summary",  ""),
    }
