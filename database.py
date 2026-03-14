"""
NutriAI - database.py
====================
Async PostgreSQL database layer using psycopg3.
Connects to Neon serverless PostgreSQL.

Tables managed here:
    users       — user profile + computed nutrition targets
    mess_menu   — one-time OCR menu setup (per institution/user)
    meal_logs   — daily food entries per user
    (chat_*     — managed automatically by LangGraph AsyncPostgresSaver)

All async functions use psycopg3 (psycopg[binary]).
Sync wrappers provided for LangChain tools which run in threads.

Install:
    pip install psycopg[binary] --break-system-packages

Environment variables (.env):
    DATABASE_URL=postgresql://neondb_owner:****@ep-...neon.tech/neondb?sslmode=require&channel_binding=require
"""

from __future__ import annotations

import os
import json
import logging
import asyncio
from datetime import date, timedelta
from typing import Optional

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("nutriai.database")

DATABASE_URL = os.getenv("DATABASE_URL")


# ══════════════════════════════════════════════════════════════════════════════
# CONNECTION
# ══════════════════════════════════════════════════════════════════════════════
async def get_connection() -> psycopg.AsyncConnection:
    """
    Open and return an async psycopg3 connection to Neon PostgreSQL.
    Neon requires sslmode=require — already present in DATABASE_URL.
    Uses dict_row so all fetchone/fetchall return dicts instead of tuples.
    """
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set in .env")
    return await psycopg.AsyncConnection.connect(
        DATABASE_URL,
        row_factory=dict_row,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA SETUP
# Called once at app startup to create tables if they don't exist.
# ══════════════════════════════════════════════════════════════════════════════
async def create_tables() -> None:
    """
    Create all NutriAI tables in Neon PostgreSQL.
    Safe to call multiple times — uses IF NOT EXISTS.
    Call this from FastAPI startup event.
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cur:

            # ── users ──────────────────────────────────────────────────────────
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id         TEXT PRIMARY KEY,
                    name            TEXT,
                    age             INTEGER,
                    gender          TEXT,
                    height_cm       REAL,
                    weight_kg       REAL,
                    goal            TEXT,           -- lose | gain | maintain
                    diet            TEXT,           -- veg | non_veg | egg
                    eats_in_mess    TEXT,           -- yes | no | mixed
                    activities      TEXT[],
                    gym_days        INTEGER DEFAULT 0,
                    gym_type        TEXT,
                    sleep_hours     REAL DEFAULT 7,
                    target_weight   REAL,
                    duration_weeks  INTEGER,

                    -- Computed targets (from onboarding agent)
                    calories        INTEGER,
                    gym_day_calories INTEGER,
                    rest_day_calories INTEGER,
                    protein_g       INTEGER,
                    carbs_g         INTEGER,
                    fats_g          INTEGER,
                    bmr             INTEGER,
                    tdee            INTEGER,
                    bmi             REAL,

                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # ── mess_menu ──────────────────────────────────────────────────────
            # One row per dish per meal slot.
            # institution_id allows multiple messes (future-proof).
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS mess_menu (
                    id              SERIAL PRIMARY KEY,
                    institution_id  TEXT NOT NULL DEFAULT 'default',
                    meal_slot       TEXT NOT NULL,  -- breakfast|lunch|snacks|dinner
                    dish_key        TEXT NOT NULL,  -- normalized dish name
                    display_name    TEXT NOT NULL,
                    calories        REAL,
                    protein_g       REAL,
                    carbs_g         REAL,
                    fats_g          REAL,
                    serving_desc    TEXT,           -- "1 cup", "2 pieces", etc.
                    portion_g       REAL,
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(institution_id, meal_slot, dish_key)
                )
            """)

            # ── meal_logs ──────────────────────────────────────────────────────
            # One row per logged food item per user per day.
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS meal_logs (
                    id              SERIAL PRIMARY KEY,
                    user_id         TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                    log_date        DATE NOT NULL DEFAULT CURRENT_DATE,
                    meal_slot       TEXT NOT NULL,  -- breakfast|lunch|snacks|dinner
                    dish_key        TEXT NOT NULL,
                    display_name    TEXT NOT NULL,
                    calories        REAL NOT NULL DEFAULT 0,
                    protein_g       REAL NOT NULL DEFAULT 0,
                    carbs_g         REAL NOT NULL DEFAULT 0,
                    fats_g          REAL NOT NULL DEFAULT 0,
                    serving_desc    TEXT,
                    portion_multiplier REAL DEFAULT 1.0,  -- 0.7=small, 1.0=medium, 1.4=large
                    source          TEXT DEFAULT 'menu',  -- menu|scan|custom
                    skipped         BOOLEAN DEFAULT FALSE,
                    notes           TEXT,
                    logged_at       TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # ── indexes for fast daily/weekly queries ──────────────────────────
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_meal_logs_user_date
                ON meal_logs(user_id, log_date)
            """)
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_meal_logs_user_slot
                ON meal_logs(user_id, log_date, meal_slot)
            """)

        await conn.commit()
    log.info("Database tables created / verified ✅")


# ══════════════════════════════════════════════════════════════════════════════
# USERS
# ══════════════════════════════════════════════════════════════════════════════
async def save_user_profile(user_id: str, profile: dict, plan: dict) -> None:
    """
    Save or update a user's profile and computed nutrition targets.
    Called at the end of onboarding when the agent computes the plan.

    Args:
        user_id : Unique user identifier
        profile : Raw profile fields from onboarding (name, age, goal, etc.)
        plan    : Computed targets from _compute_plan() in agent.py
    """
    activities = profile.get("activities", [])
    if isinstance(activities, str):
        activities = [a.strip() for a in activities.split(",")]

    async with await get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO users (
                    user_id, name, age, gender, height_cm, weight_kg,
                    goal, diet, eats_in_mess, activities, gym_days, gym_type,
                    sleep_hours, target_weight, duration_weeks,
                    calories, gym_day_calories, rest_day_calories,
                    protein_g, carbs_g, fats_g, bmr, tdee, bmi,
                    updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    NOW()
                )
                ON CONFLICT (user_id) DO UPDATE SET
                    name              = EXCLUDED.name,
                    age               = EXCLUDED.age,
                    gender            = EXCLUDED.gender,
                    height_cm         = EXCLUDED.height_cm,
                    weight_kg         = EXCLUDED.weight_kg,
                    goal              = EXCLUDED.goal,
                    diet              = EXCLUDED.diet,
                    eats_in_mess      = EXCLUDED.eats_in_mess,
                    activities        = EXCLUDED.activities,
                    gym_days          = EXCLUDED.gym_days,
                    gym_type          = EXCLUDED.gym_type,
                    sleep_hours       = EXCLUDED.sleep_hours,
                    target_weight     = EXCLUDED.target_weight,
                    duration_weeks    = EXCLUDED.duration_weeks,
                    calories          = EXCLUDED.calories,
                    gym_day_calories  = EXCLUDED.gym_day_calories,
                    rest_day_calories = EXCLUDED.rest_day_calories,
                    protein_g         = EXCLUDED.protein_g,
                    carbs_g           = EXCLUDED.carbs_g,
                    fats_g            = EXCLUDED.fats_g,
                    bmr               = EXCLUDED.bmr,
                    tdee              = EXCLUDED.tdee,
                    bmi               = EXCLUDED.bmi,
                    updated_at        = NOW()
            """, (
                user_id,
                profile.get("name"),
                profile.get("age"),
                profile.get("gender"),
                profile.get("height"),
                profile.get("weight"),
                plan.get("goal"),
                profile.get("diet"),
                str(profile.get("eats_in_mess", "yes")),
                activities,
                plan.get("gym_days_per_week", 0),
                profile.get("gym_type"),
                profile.get("sleep", 7),
                profile.get("target_weight"),
                profile.get("duration"),
                plan.get("calories"),
                plan.get("gymDayCalories"),
                plan.get("restDayCalories"),
                plan.get("protein"),
                plan.get("carbs"),
                plan.get("fats"),
                plan.get("bmr"),
                plan.get("tdee"),
                float(plan.get("bmi", 0)),
            ))
        await conn.commit()
    log.info(f"User profile saved for '{user_id}' ✅")


async def get_user_profile(user_id: str) -> Optional[dict]:
    """
    Fetch a user's full profile + targets from PostgreSQL.
    Called by chatbot at session start to build the system prompt.

    Returns None if user not found (not yet onboarded).
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT * FROM users WHERE user_id = %s",
                (user_id,)
            )
            row = await cur.fetchone()
            return dict(row) if row else None


# ══════════════════════════════════════════════════════════════════════════════
# MESS MENU
# ══════════════════════════════════════════════════════════════════════════════
async def save_mess_menu(
    dishes         : list[dict],
    institution_id : str = "default",
) -> int:
    """
    Save the full mess menu to PostgreSQL after OCR setup.
    Replaces existing menu for the institution (upsert on conflict).

    Args:
        dishes         : List of dish dicts with nutrition info
        institution_id : Institution identifier (default = "default")

    Returns:
        Number of dishes saved.
    """
    if not dishes:
        return 0

    async with await get_connection() as conn:
        async with conn.cursor() as cur:
            count = 0
            for dish in dishes:
                await cur.execute("""
                    INSERT INTO mess_menu (
                        institution_id, meal_slot, dish_key, display_name,
                        calories, protein_g, carbs_g, fats_g,
                        serving_desc, portion_g
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (institution_id, meal_slot, dish_key)
                    DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        calories     = EXCLUDED.calories,
                        protein_g    = EXCLUDED.protein_g,
                        carbs_g      = EXCLUDED.carbs_g,
                        fats_g       = EXCLUDED.fats_g,
                        serving_desc = EXCLUDED.serving_desc,
                        portion_g    = EXCLUDED.portion_g
                """, (
                    institution_id,
                    dish.get("meal_slot", "lunch"),
                    dish.get("dish_key", dish.get("dish", "")),
                    dish.get("display_name", dish.get("dish", "").replace("_", " ").title()),
                    dish.get("calories"),
                    dish.get("protein"),
                    dish.get("carbs"),
                    dish.get("fats"),
                    dish.get("serving_desc"),
                    dish.get("portion_g"),
                ))
                count += 1
        await conn.commit()
    log.info(f"Mess menu saved — {count} dishes for '{institution_id}' ✅")
    return count


async def get_mess_menu(
    institution_id : str = "default",
    meal_slot      : Optional[str] = None,
) -> list[dict]:
    """
    Fetch the mess menu from PostgreSQL.

    Args:
        institution_id : Institution identifier
        meal_slot      : Filter by slot (breakfast|lunch|snacks|dinner).
                         If None, returns all slots.

    Returns:
        List of dish dicts grouped by meal_slot.
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cur:
            if meal_slot:
                await cur.execute("""
                    SELECT * FROM mess_menu
                    WHERE institution_id = %s AND meal_slot = %s
                    ORDER BY meal_slot, display_name
                """, (institution_id, meal_slot))
            else:
                await cur.execute("""
                    SELECT * FROM mess_menu
                    WHERE institution_id = %s
                    ORDER BY meal_slot, display_name
                """, (institution_id,))

            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def get_mess_menu_grouped(institution_id: str = "default") -> dict:
    """
    Fetch menu grouped by meal slot.
    Returns: { "breakfast": [...], "lunch": [...], "snacks": [...], "dinner": [...] }
    """
    rows = await get_mess_menu(institution_id)
    grouped: dict = {"breakfast": [], "lunch": [], "snacks": [], "dinner": []}
    for row in rows:
        slot = row.get("meal_slot", "lunch")
        if slot in grouped:
            grouped[slot].append(row)
    return grouped


# ══════════════════════════════════════════════════════════════════════════════
# MEAL LOGS
# ══════════════════════════════════════════════════════════════════════════════
async def log_meal(
    user_id    : str,
    meal_slot  : str,
    dish       : dict,
    log_date   : Optional[date] = None,
    source     : str = "menu",
) -> int:
    """
    Log a single meal item for a user.

    Args:
        user_id   : User identifier
        meal_slot : breakfast | lunch | snacks | dinner
        dish      : Dict with nutrition info (dish_key, calories, protein, etc.)
        log_date  : Date to log for (defaults to today)
        source    : How it was added — menu | scan | custom

    Returns:
        ID of the inserted log row.
    """
    log_date = log_date or date.today()
    multiplier = dish.get("portion_multiplier", 1.0)

    async with await get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO meal_logs (
                    user_id, log_date, meal_slot,
                    dish_key, display_name,
                    calories, protein_g, carbs_g, fats_g,
                    serving_desc, portion_multiplier, source,
                    skipped, notes
                ) VALUES (
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s
                )
                RETURNING id
            """, (
                user_id, log_date, meal_slot,
                dish.get("dish_key", dish.get("dish", "")),
                dish.get("display_name", dish.get("dish", "").replace("_", " ").title()),
                round(dish.get("calories", 0) * multiplier, 1),
                round(dish.get("protein",  0) * multiplier, 1),
                round(dish.get("carbs",    0) * multiplier, 1),
                round(dish.get("fats",     0) * multiplier, 1),
                dish.get("serving_desc"),
                multiplier,
                source,
                dish.get("skipped", False),
                dish.get("notes"),
            ))
            row = await cur.fetchone()
        await conn.commit()

    entry_id = row["id"]
    log.info(f"Meal logged — user={user_id} slot={meal_slot} dish={dish.get('dish_key')} id={entry_id}")
    return entry_id


async def delete_meal_log(log_id: int, user_id: str) -> bool:
    """Delete a specific meal log entry. user_id check prevents cross-user deletion."""
    async with await get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM meal_logs WHERE id = %s AND user_id = %s",
                (log_id, user_id)
            )
            deleted = cur.rowcount > 0
        await conn.commit()
    return deleted


async def get_daily_logs(
    user_id  : str,
    log_date : Optional[date] = None,
) -> list[dict]:
    """
    Fetch all meal log entries for a user on a given date.
    Defaults to today.
    """
    log_date = log_date or date.today()
    async with await get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT * FROM meal_logs
                WHERE user_id = %s AND log_date = %s AND skipped = FALSE
                ORDER BY logged_at ASC
            """, (user_id, log_date))
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def get_today_totals(user_id: str) -> dict:
    """
    Compute today's macro totals for a user.
    Joins with users table to include targets.

    Returns dict with calories, protein_g, carbs_g, fats_g
    plus target_calories, target_protein from user profile.
    """
    today = date.today()
    async with await get_connection() as conn:
        async with conn.cursor() as cur:
            # Get today's totals
            await cur.execute("""
                SELECT
                    COALESCE(SUM(calories),  0) AS calories,
                    COALESCE(SUM(protein_g), 0) AS protein,
                    COALESCE(SUM(carbs_g),   0) AS carbs,
                    COALESCE(SUM(fats_g),    0) AS fats,
                    COUNT(*) AS items_logged
                FROM meal_logs
                WHERE user_id = %s
                  AND log_date = %s
                  AND skipped = FALSE
            """, (user_id, today))
            totals = dict(await cur.fetchone())

            # Get user targets
            await cur.execute("""
                SELECT calories, protein_g, carbs_g, fats_g
                FROM users WHERE user_id = %s
            """, (user_id,))
            targets = await cur.fetchone()

    if targets:
        totals["target_calories"] = targets["calories"] or 2000
        totals["target_protein"]  = targets["protein_g"] or 120
        totals["target_carbs"]    = targets["carbs_g"] or 250
        totals["target_fats"]     = targets["fats_g"] or 55
    else:
        totals["target_calories"] = 2000
        totals["target_protein"]  = 120
        totals["target_carbs"]    = 250
        totals["target_fats"]     = 55

    totals["remaining_calories"] = max(0, totals["target_calories"] - totals["calories"])
    totals["remaining_protein"]  = max(0, totals["target_protein"]  - totals["protein"])
    totals["date"] = today.isoformat()
    return totals


async def get_weekly_summary(user_id: str) -> dict:
    """
    Compute 7-day macro averages and consistency stats for a user.
    Used by agent.py weekly review and chatbot get_weekly_summary tool.
    """
    today = date.today()
    week_ago = today - timedelta(days=6)

    async with await get_connection() as conn:
        async with conn.cursor() as cur:
            # Per-day totals for last 7 days
            await cur.execute("""
                SELECT
                    log_date,
                    SUM(calories)  AS cal,
                    SUM(protein_g) AS pro,
                    SUM(carbs_g)   AS car,
                    SUM(fats_g)    AS fat,
                    COUNT(*)       AS items
                FROM meal_logs
                WHERE user_id = %s
                  AND log_date BETWEEN %s AND %s
                  AND skipped = FALSE
                GROUP BY log_date
                ORDER BY log_date ASC
            """, (user_id, week_ago, today))
            per_day = {row["log_date"].isoformat(): dict(row)
                       for row in await cur.fetchall()}

            # User targets
            await cur.execute(
                "SELECT calories, protein_g FROM users WHERE user_id = %s",
                (user_id,)
            )
            targets = await cur.fetchone()

    cal_target = targets["calories"]  if targets else 2000
    pro_target = targets["protein_g"] if targets else 120

    tracked = [d for d in per_day.values() if d["items"] > 0]
    n = len(tracked)

    if n == 0:
        return {
            "n_tracked"      : 0,
            "per_day"        : per_day,
            "avg_calories"   : 0,
            "avg_protein"    : 0,
            "target_calories": cal_target,
            "target_protein" : pro_target,
            "cal_hit_days"   : 0,
            "pro_hit_days"   : 0,
        }

    avg_cal = round(sum(d["cal"] for d in tracked) / n)
    avg_pro = round(sum(d["pro"] for d in tracked) / n)

    cal_hit = sum(1 for d in tracked if abs(d["cal"] - cal_target) < cal_target * 0.1)
    pro_hit = sum(1 for d in tracked if d["pro"] >= pro_target * 0.9)

    return {
        "n_tracked"      : n,
        "per_day"        : per_day,
        "avg_calories"   : avg_cal,
        "avg_protein"    : avg_pro,
        "target_calories": cal_target,
        "target_protein" : pro_target,
        "cal_hit_days"   : cal_hit,
        "pro_hit_days"   : pro_hit,
        "cal_hit_pct"    : round(cal_hit / n * 100),
        "pro_hit_pct"    : round(pro_hit / n * 100),
    }


async def get_date_range_logs(
    user_id   : str,
    start_date: date,
    end_date  : date,
) -> dict:
    """
    Fetch raw meal logs for a date range.
    Used by agent.py gap analysis — compare planned vs actually eaten.

    Returns: { "YYYY-MM-DD": [meal_log_rows], ... }
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                SELECT * FROM meal_logs
                WHERE user_id = %s
                  AND log_date BETWEEN %s AND %s
                ORDER BY log_date, meal_slot, logged_at
            """, (user_id, start_date, end_date))
            rows = await cur.fetchall()

    grouped: dict = {}
    for row in rows:
        dk = row["log_date"].isoformat()
        if dk not in grouped:
            grouped[dk] = []
        grouped[dk].append(dict(row))
    return grouped


# ══════════════════════════════════════════════════════════════════════════════
# SYNC WRAPPERS
# LangChain tools run in threads — they need sync functions.
# These wrappers use asyncio.run() to call async functions safely.
# ══════════════════════════════════════════════════════════════════════════════
def get_today_totals_sync(user_id: str) -> dict:
    """Sync wrapper for get_today_totals — used by LangChain tools in chatbot.py."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In async context (FastAPI) — create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, get_today_totals(user_id))
                return future.result(timeout=10)
        else:
            return loop.run_until_complete(get_today_totals(user_id))
    except Exception as e:
        log.error(f"get_today_totals_sync failed: {e}")
        return {}


def get_weekly_summary_sync(user_id: str) -> dict:
    """Sync wrapper for get_weekly_summary — used by LangChain tools in chatbot.py."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, get_weekly_summary(user_id))
                return future.result(timeout=10)
        else:
            return loop.run_until_complete(get_weekly_summary(user_id))
    except Exception as e:
        log.error(f"get_weekly_summary_sync failed: {e}")
        return {}
