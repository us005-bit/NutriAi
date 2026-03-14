"""
NutriAI - chatbot.py
==================
Context-aware, persistent nutrition chatbot built with LangGraph.

Architecture:
    LangGraph StateGraph with tool-calling loop:
        load_context → should_tool → [tool_node | generate] → END

Memory layers:
    Short-term : LangGraph message state (current conversation)
    Long-term  : PostgreSQL via AsyncPostgresSaver (persists across sessions)
    Semantic   : RAG via rag.py (nutrition_db + mess menu + user log)

Tools the agent can call:
    get_nutrition      → look up any dish's macros via nutrition.py 5-layer pipeline
    get_daily_totals   → fetch today's logged meals + running macro totals from DB
    get_weekly_summary → fetch 7-day log summary for context

LangGraph concepts demonstrated (CampusX playlist):
    Video 3  — LLM (HuggingFaceHub / ChatHuggingFace)
    Video 4  — ChatPromptTemplate
    Video 6  — StrOutputParser
    Video 8  — Runnables
    Video 16 — Tools
    Video 17 — Tool calling
    Video 18 — End-to-end agent
    LG-4     — StateGraph, nodes, edges
    LG-5     — Sequential workflow
    LG-7     — Conditional workflow (tool router)
    LG-8     — Iterative workflow (tool loop)
    LG-9     — Chatbot with persistence
    LG-11    — Streaming
    LG-14    — Short + long term memory
    LG-16    — Tools in LangGraph
    LG-35    — LangSmith observability

Install:
    pip install langchain langgraph langchain-community langchain-core
                psycopg[binary] langsmith --break-system-packages

Environment variables (.env):
    HF_API_KEY=hf_xxxx
    DATABASE_URL=postgresql://neondb_owner:****@ep-...neon.tech/neondb?sslmode=require
    LANGSMITH_API_KEY=lsv2_xxxx          # optional — enables tracing
    LANGSMITH_PROJECT=nutriai              # optional
    LANGCHAIN_TRACING_V2=true            # optional
"""

from __future__ import annotations

import os
import json
import logging
from datetime import date
from typing import Annotated, Any, Optional, AsyncIterator

from dotenv import load_dotenv
load_dotenv()

# ── LangSmith observability (auto-enabled if env vars set) ────────────────────
# No code needed — LangChain reads LANGCHAIN_TRACING_V2 + LANGSMITH_API_KEY
# automatically. Every chain/tool call gets traced in LangSmith dashboard.

log = logging.getLogger("nutriai.chatbot")

# ── LangGraph imports ─────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_community.llms import HuggingFaceHub


# ══════════════════════════════════════════════════════════════════════════════
# STATE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
class ChatState(TypedDict):
    """
    LangGraph state — passed between every node.

    messages     : Full conversation history. add_messages reducer
                   appends new messages rather than replacing the list.
                   This is LangGraph's built-in short-term memory.

    user_id      : Identifies the user — used to scope DB queries.
    user_profile : Loaded from PostgreSQL at session start (long-term memory).
                   Contains goal, calorie target, protein target, name, etc.
    today_totals : Running macro totals for today (fetched via tool).
    rag_context  : Semantic context retrieved from ChromaDB for this turn.
    error        : Non-fatal error message propagated to frontend.
    """
    messages     : Annotated[list, add_messages]
    user_id      : str
    user_profile : dict
    today_totals : dict
    rag_context  : str
    error        : Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS
# Tools are functions the LLM can decide to call mid-conversation.
# LangGraph's ToolNode handles execution automatically.
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_nutrition(dish_name: str) -> str:
    """
    Look up nutrition info for any dish — calories, protein, carbs, fats.
    Uses NutriAI's 5-layer pipeline: cache → DB → fuzzy → dual-LLM → Gemini.
    Call this when the user asks about a specific food's nutrition values.
    """
    from nutrition import get_nutrition_safe
    result = get_nutrition_safe(dish_name)
    if result.get("error"):
        return f"Could not find nutrition data for '{dish_name}'."
    return (
        f"{result['dish'].replace('_', ' ').title()}: "
        f"{result['calories']} kcal, "
        f"Protein {result['protein']}g, "
        f"Carbs {result['carbs']}g, "
        f"Fats {result['fats']}g "
        f"(per {result['serving_desc'] if 'serving_desc' in result else str(result['portion_g'])+'g'})."
    )


@tool
def get_daily_totals(user_id: str) -> str:
    """
    Fetch today's logged meals and running macro totals for a user.
    Call this when the user asks how much they've eaten today,
    how many calories are left, or wants to know their current progress.
    """
    import asyncio
    from database import get_today_totals_sync
    try:
        totals = get_today_totals_sync(user_id)
        if not totals:
            return "No meals logged today yet."
        return (
            f"Today so far: {totals['calories']} kcal eaten "
            f"(target: {totals['target_calories']} kcal, "
            f"remaining: {max(0, totals['target_calories'] - totals['calories'])} kcal). "
            f"Protein: {totals['protein']}g / {totals['target_protein']}g. "
            f"Carbs: {totals['carbs']}g. Fats: {totals['fats']}g."
        )
    except Exception as e:
        log.error(f"get_daily_totals tool error: {e}")
        return "Could not fetch today's totals right now."


@tool
def get_weekly_summary(user_id: str) -> str:
    """
    Fetch a summary of the user's last 7 days — average calories,
    protein, and consistency vs targets.
    Call this when the user asks about their weekly progress,
    trends, or wants a review of how they've been eating.
    """
    from database import get_weekly_summary_sync
    try:
        summary = get_weekly_summary_sync(user_id)
        if not summary or summary.get("n_tracked", 0) == 0:
            return "No meals logged in the last 7 days."
        return (
            f"Last 7 days ({summary['n_tracked']} days tracked): "
            f"Average {summary['avg_calories']} kcal/day "
            f"(target {summary['target_calories']}), "
            f"Average protein {summary['avg_protein']}g/day "
            f"(target {summary['target_protein']}g). "
            f"Hit calorie target {summary['cal_hit_days']}/{summary['n_tracked']} days."
        )
    except Exception as e:
        log.error(f"get_weekly_summary tool error: {e}")
        return "Could not fetch weekly summary right now."


# All tools the agent can use
TOOLS = [get_nutrition, get_daily_totals, get_weekly_summary]


# ══════════════════════════════════════════════════════════════════════════════
# LLM SETUP
# ══════════════════════════════════════════════════════════════════════════════
def _get_llm():
    """
    Return HuggingFace LLM with tool binding.
    Uses Qwen 72B as primary — best for nutrition + Indian food context.
    """
    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        raise RuntimeError("HF_API_KEY not set in .env")

    llm = HuggingFaceHub(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        huggingfacehub_api_token=hf_key,
        model_kwargs={
            "temperature": 0.4,
            "max_new_tokens": 400,
        },
    )
    # Bind tools so LLM knows what it can call
    return llm.bind_tools(TOOLS)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════
def _build_system_prompt(user_profile: dict, rag_context: str) -> str:
    """
    Build the system prompt dynamically from user profile + RAG context.
    This is injected at the start of every conversation turn.
    """
    name       = user_profile.get("name", "the user")
    goal       = user_profile.get("goal", "maintain")
    cal_target = user_profile.get("calories", 2000)
    pro_target = user_profile.get("protein", 120)
    diet       = user_profile.get("diet", "non_veg")
    gym_days   = user_profile.get("gym_days_per_week", 0)

    goal_desc = {
        "lose"    : "losing weight (calorie deficit)",
        "gain"    : "building muscle (calorie surplus, high protein)",
        "maintain": "maintaining weight (balanced macros)",
    }.get(goal, "balanced nutrition")

    gym_note = f"They go to the gym {gym_days} days/week." if gym_days > 0 else ""

    return f"""You are NutriAI, a friendly and knowledgeable nutrition assistant for Indian college students.

USER PROFILE:
- Name: {name}
- Goal: {goal_desc}
- Daily targets: {cal_target} kcal, {pro_target}g protein
- Diet preference: {diet}
- {gym_note}

RELEVANT NUTRITION CONTEXT (from NutriAI knowledge base):
{rag_context if rag_context else "No specific context retrieved for this query."}

INSTRUCTIONS:
- Be conversational, warm, and specific. Address the user by name occasionally.
- Always ground advice in their actual targets and today's intake when relevant.
- For nutrition questions use the get_nutrition tool to get accurate values.
- For today's progress use get_daily_totals tool.
- For weekly trends use get_weekly_summary tool.
- Give practical mess-food advice — suggest real dishes available in Indian college messes.
- Keep responses concise (3-6 sentences) unless the user asks for detail.
- Never make up nutrition values — use tools or RAG context.
- Today's date: {date.today().isoformat()}"""


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def node_load_context(state: ChatState) -> ChatState:
    """
    Node 1: Load RAG context for the latest user message.
    Retrieves semantically relevant nutrition chunks from ChromaDB.
    This runs before every LLM call — keeps context fresh per turn.
    """
    from rag import retrieve_context

    # Get the latest user message
    last_human = next(
        (m.content for m in reversed(state["messages"])
         if isinstance(m, HumanMessage)),
        ""
    )

    if not last_human:
        return {**state, "rag_context": ""}

    try:
        chunks = retrieve_context(last_human, k=4)
        rag_context = "\n".join(chunks) if chunks else ""
    except Exception as e:
        log.warning(f"RAG retrieval failed (non-fatal): {e}")
        rag_context = ""

    log.info(f"RAG: retrieved {len(chunks) if chunks else 0} chunks for context")
    return {**state, "rag_context": rag_context}


def node_generate(state: ChatState) -> ChatState:
    """
    Node 2: Generate LLM response with full context.

    Builds system prompt from user profile + RAG context,
    then runs the LLM with full message history (short-term memory)
    and tool binding.
    """
    llm = _get_llm()

    # Build dynamic system prompt
    system_content = _build_system_prompt(
        state["user_profile"],
        state["rag_context"],
    )

    # Full message list: system + conversation history
    messages = [SystemMessage(content=system_content)] + state["messages"]

    try:
        response = llm.invoke(messages)
    except Exception as e:
        log.error(f"LLM generation failed: {e}")
        response = AIMessage(content=(
            "I'm having trouble connecting right now. "
            "Please try again in a moment."
        ))

    return {**state, "messages": [response]}


def node_update_totals(state: ChatState) -> ChatState:
    """
    Node 3: After tool calls, refresh today's totals in state.
    Ensures the next LLM response has up-to-date consumption data.
    """
    from database import get_today_totals_sync
    try:
        totals = get_today_totals_sync(state["user_id"])
        return {**state, "today_totals": totals or {}}
    except Exception as e:
        log.warning(f"Totals refresh failed (non-fatal): {e}")
        return state


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ══════════════════════════════════════════════════════════════════════════════
def should_use_tool(state: ChatState) -> str:
    """
    Conditional edge — router between tool call and final response.

    If the last AI message contains tool_calls → route to tool_node.
    Otherwise → END (response is ready).

    This implements the iterative tool loop from LG-8.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        log.info(f"Routing to tools: {[t['name'] for t in last_message.tool_calls]}")
        return "tools"
    return "end"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
def build_chatbot_graph(checkpointer=None):
    """
    Build the NutriAI chatbot LangGraph StateGraph.

    Graph flow:
        load_context
            → generate
                → [tool_calls?]
                    YES → tools → update_totals → generate  (loop)
                    NO  → END

    Args:
        checkpointer: AsyncPostgresSaver instance for persistence.
                      If None, graph runs without persistence (dev mode).

    Demonstrates:
        - StateGraph with typed state          (LG-4)
        - Sequential edges                     (LG-5)
        - Conditional edges / routing          (LG-7)
        - Iterative tool loop                  (LG-8)
        - PostgreSQL checkpointer persistence  (LG-9)
        - Prebuilt ToolNode                    (LG-16)
    """
    g = StateGraph(ChatState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    g.add_node("load_context",   node_load_context)
    g.add_node("generate",       node_generate)
    g.add_node("tools",          ToolNode(TOOLS))       # LangGraph prebuilt
    g.add_node("update_totals",  node_update_totals)

    # ── Edges ──────────────────────────────────────────────────────────────────
    g.set_entry_point("load_context")
    g.add_edge("load_context", "generate")

    # Conditional: did the LLM decide to call a tool?
    g.add_conditional_edges(
        "generate",
        should_use_tool,
        {"tools": "tools", "end": END},
    )

    # After tool execution → refresh totals → generate again (loop)
    g.add_edge("tools",         "update_totals")
    g.add_edge("update_totals", "generate")

    # ── Compile with optional checkpointer ────────────────────────────────────
    if checkpointer:
        graph = g.compile(checkpointer=checkpointer)
        log.info("Chatbot graph compiled with PostgreSQL persistence ✅")
    else:
        graph = g.compile()
        log.warning("Chatbot graph compiled WITHOUT persistence (dev mode)")

    return graph


# ══════════════════════════════════════════════════════════════════════════════
# POSTGRESQL CHECKPOINTER SETUP
# ══════════════════════════════════════════════════════════════════════════════
async def get_checkpointer():
    """
    Create and return AsyncPostgresSaver checkpointer.
    Called once at app startup.

    Uses psycopg3 async driver — matches LangGraph's recommendation
    and your backend teammate's pg library choice.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    import psycopg

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set in .env")

    # Neon requires sslmode=require — already in the connection string
    # autocommit=True is required by AsyncPostgresSaver — without it the
    # connection stays in an aborted transaction state after any DB error
    # during startup (e.g. create_tables), which causes the chatbot graph
    # init to silently fail with "current transaction is aborted".
    conn = await psycopg.AsyncConnection.connect(db_url, autocommit=True)
    checkpointer = AsyncPostgresSaver(conn)

    # Create LangGraph checkpoint tables if they don't exist
    await checkpointer.setup()
    log.info("PostgreSQL checkpointer ready ✅")
    return checkpointer


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON GRAPH
# ══════════════════════════════════════════════════════════════════════════════
_graph = None

async def get_graph():
    """
    Return compiled chatbot graph (singleton).
    Initializes PostgreSQL checkpointer on first call.
    """
    global _graph
    if _graph is None:
        checkpointer = await get_checkpointer()
        _graph = build_chatbot_graph(checkpointer=checkpointer)
    return _graph


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
async def chat(
    user_id    : str,
    message    : str,
    user_profile: dict,
    thread_id  : Optional[str] = None,
) -> str:
    """
    Send a message and get a full response (non-streaming).

    Args:
        user_id      : User's unique ID (from PostgreSQL users table)
        message      : User's message text
        user_profile : User's profile dict (from PostgreSQL)
        thread_id    : Conversation thread ID. If None, uses user_id.
                       Pass different thread_ids for separate conversations.

    Returns:
        Assistant's response as a string.

    PostgreSQL persistence means conversation history is automatically
    loaded and saved — user can continue from where they left off
    even after closing the app.
    """
    graph     = await get_graph()
    thread_id = thread_id or user_id

    # LangGraph config — thread_id scopes the checkpointed conversation
    config = {"configurable": {"thread_id": thread_id}}

    # Initial state for this turn
    # (LangGraph loads previous messages from PostgreSQL automatically)
    state = {
        "messages"    : [HumanMessage(content=message)],
        "user_id"     : user_id,
        "user_profile": user_profile,
        "today_totals": {},
        "rag_context" : "",
        "error"       : None,
    }

    result = await graph.ainvoke(state, config=config)

    # Extract last AI message
    last_ai = next(
        (m for m in reversed(result["messages"])
         if isinstance(m, AIMessage) and not m.tool_calls),
        None
    )

    return last_ai.content if last_ai else "Sorry, I couldn't generate a response."


async def stream_chat(
    user_id     : str,
    message     : str,
    user_profile: dict,
    thread_id   : Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Send a message and stream the response token by token.
    Use this for the frontend chat UI — yields tokens as they arrive.

    Usage in FastAPI:
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            stream_chat(user_id, message, profile),
            media_type="text/event-stream"
        )

    Demonstrates LangGraph streaming (LG-11).
    """
    graph     = await get_graph()
    thread_id = thread_id or user_id
    config    = {"configurable": {"thread_id": thread_id}}

    state = {
        "messages"    : [HumanMessage(content=message)],
        "user_id"     : user_id,
        "user_profile": user_profile,
        "today_totals": {},
        "rag_context" : "",
        "error"       : None,
    }

    # astream_events yields fine-grained events including LLM tokens
    async for event in graph.astream_events(state, config=config, version="v2"):
        kind = event["event"]

        # Yield text tokens as they stream from the LLM
        if kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content

        # Yield tool call notifications so frontend can show "Looking up..."
        elif kind == "on_tool_start":
            tool_name = event.get("name", "tool")
            friendly  = {
                "get_nutrition"      : "Looking up nutrition info...",
                "get_daily_totals"   : "Checking today's intake...",
                "get_weekly_summary" : "Fetching weekly summary...",
            }.get(tool_name, f"Using {tool_name}...")
            yield f"\n[{friendly}]\n"


async def get_chat_history(
    user_id  : str,
    thread_id: Optional[str] = None,
    limit    : int = 20,
) -> list[dict]:
    """
    Fetch recent chat history for a user from PostgreSQL checkpointer.

    Returns list of { role, content } dicts for frontend display.
    This is the long-term memory retrieval — user sees their full
    conversation history even after closing the app.
    """
    graph     = await get_graph()
    thread_id = thread_id or user_id
    config    = {"configurable": {"thread_id": thread_id}}

    try:
        state = await graph.aget_state(config)
        if not state or not state.values:
            return []

        messages = state.values.get("messages", [])
        history  = []

        for m in messages[-limit:]:
            if isinstance(m, HumanMessage):
                history.append({"role": "user",      "content": m.content})
            elif isinstance(m, AIMessage) and not m.tool_calls:
                history.append({"role": "assistant", "content": m.content})

        return history

    except Exception as e:
        log.error(f"Failed to fetch chat history: {e}")
        return []


async def clear_chat_history(
    user_id  : str,
    thread_id: Optional[str] = None,
) -> bool:
    """
    Clear conversation history for a thread.
    Called when user taps 'New conversation' in the UI.
    """
    # LangGraph doesn't expose delete directly on the checkpointer
    # We start a fresh thread by generating a new thread_id on the frontend
    # This function is a placeholder for future direct DB deletion
    log.info(f"Chat history clear requested for user {user_id}, thread {thread_id}")
    return True