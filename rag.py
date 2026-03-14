"""
NutriAI - rag.py
==============
Retrieval-Augmented Generation for nutrition Q&A.
Rebuilt using LangChain LCEL pipeline.

Pipeline:
    User question
        → LangChain HuggingFaceEmbeddings
        → ChromaDB vector store (LangChain wrapper)
        → LangChain Retriever (top-k similarity)
        → ChatPromptTemplate
        → ChatHuggingFace LLM
        → StrOutputParser
        → answer

Document sources indexed:
    1. nutrition_db.py  → one chunk per dish (name + macros + health tags)
    2. User's mess menu → upserted at setup time (one-time + on update)
    3. User's meal log  → upserted per user_id at query time

Install:
    pip install langchain langchain-community langchain-chroma
                chromadb sentence-transformers --break-system-packages

Environment variables (.env):
    HF_API_KEY=hf_xxxx

Usage:
    from rag import build_rag_chain, upsert_menu, upsert_user_log
    chain  = build_rag_chain()
    answer = chain.invoke({
        "question" : "Is today's lunch good for weight loss?",
        "user_goal": "lose",
    })
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger("nutriai.rag")

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_DIR   = Path(__file__).parent / ".chroma_db"
COLLECTION   = "nutriai_nutrition"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # 80MB, fast, cosine similarity
TOP_K        = 4                     # chunks retrieved per query
MAX_CONTEXT  = 1200                  # chars passed to LLM

# ── Module-level singletons ────────────────────────────────────────────────────
_vectorstore      = None
_retriever        = None
_rag_chain        = None
_db_populated     = False
_last_menu_hash   = None
_last_log_hashes  = {}   # { user_id: md5_hash }


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDINGS — LangChain HuggingFaceEmbeddings
# ══════════════════════════════════════════════════════════════════════════════
def _get_embeddings():
    """Return LangChain HuggingFace embedding model (singleton)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ══════════════════════════════════════════════════════════════════════════════
# VECTOR STORE — LangChain ChromaDB wrapper
# ══════════════════════════════════════════════════════════════════════════════
def _get_vectorstore():
    """Return LangChain ChromaDB vectorstore (singleton)."""
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma
        _vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=_get_embeddings(),
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
        log.info(f"ChromaDB vectorstore loaded — {_vectorstore._collection.count()} docs")
    return _vectorstore


def _get_retriever():
    """Return LangChain retriever (singleton)."""
    global _retriever
    if _retriever is None:
        vs = _get_vectorstore()
        _retriever = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K},
        )
        log.info(f"Retriever ready (top_k={TOP_K})")
    return _retriever


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def _dish_to_chunk(dish_key: str, data: dict) -> str:
    """
    Convert a nutrition_db entry into a rich text chunk for embedding.
    Health tags make retrieval smarter — 'high protein' queries find
    relevant dishes even if the dish name isn't mentioned.
    """
    name = dish_key.replace("_", " ").title()
    cal  = data.get("calories",     0)
    pro  = data.get("protein",      0)
    car  = data.get("carbs",        0)
    fat  = data.get("fats",         0)
    pg   = data.get("portion_g",    0)
    desc = data.get("serving_desc", "1 serving")

    tags = []
    if pro >= 12:  tags.append("high protein")
    if cal <= 150: tags.append("low calorie")
    if fat <= 3:   tags.append("low fat")
    if car >= 40:  tags.append("high carb")
    if fat >= 15:  tags.append("high fat")
    tag_str = ", ".join(tags) if tags else "moderate nutrition"

    return (
        f"{name}: {cal} kcal per serving ({desc}, {pg}g). "
        f"Protein {pro}g, Carbs {car}g, Fats {fat}g. "
        f"Profile: {tag_str}."
    )


def _menu_to_documents(menu: dict):
    """Convert mess menu dict into LangChain Documents."""
    from langchain_core.documents import Document
    docs = []
    for meal_id, dishes in menu.items():
        for d in (dishes or []):
            if not d.get("name"):
                continue
            name = d["name"].replace("_", " ").title()
            # Natural serving unit from nutrition_db if available
            serving = d.get("serving_desc", "1 serving")
            text = (
                f"Mess {meal_id} has {name}: "
                f"{round(d.get('calories', 0))} kcal, "
                f"{d.get('protein', 0):.1f}g protein, "
                f"{d.get('carbs', 0):.1f}g carbs, "
                f"{d.get('fats', 0):.1f}g fats. "
                f"Serving: {serving}."
            )
            docs.append(Document(
                page_content=text,
                metadata={
                    "source"  : "menu",
                    "meal_id" : meal_id,
                    "dish"    : d["name"],
                }
            ))
    return docs


def _log_to_documents(user_log: dict, user_id: str):
    """Convert user's meal log (last 7 days) into LangChain Documents."""
    from langchain_core.documents import Document
    docs = []
    for date_key, day_data in list(user_log.items())[-7:]:
        for meal_id, items in day_data.items():
            for item in (items or []):
                if item.get("skipped") or not item.get("name"):
                    continue
                name = item["name"].replace("_", " ").title()
                text = (
                    f"On {date_key} at {meal_id}, user {user_id} ate {name}: "
                    f"{round(item.get('calories', 0))} kcal, "
                    f"{item.get('protein', 0):.1f}g protein, "
                    f"{item.get('carbs', 0):.1f}g carbs, "
                    f"{item.get('fats', 0):.1f}g fats."
                )
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source" : "user_log",
                        "user_id": user_id,
                        "date"   : date_key,
                        "meal_id": meal_id,
                    }
                ))
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# POPULATION
# ══════════════════════════════════════════════════════════════════════════════
def populate_nutrition_db(force: bool = False) -> int:
    """
    Index all nutrition_db dishes into ChromaDB.
    Runs once on startup — skips if already populated unless force=True.

    Returns number of documents in collection after population.
    """
    global _db_populated
    if _db_populated and not force:
        return 0

    vs  = _get_vectorstore()
    col = vs._collection

    if col.count() > 0 and not force:
        log.info(f"ChromaDB already has {col.count()} docs — skipping population.")
        _db_populated = True
        return col.count()

    from nutrition_db import NUTRITION_DB
    from langchain_core.documents import Document

    log.info(f"Indexing {len(NUTRITION_DB)} nutrition_db dishes into ChromaDB…")

    docs = []
    ids  = []
    for dish_key, data in NUTRITION_DB.items():
        chunk = _dish_to_chunk(dish_key, data)
        docs.append(Document(
            page_content=chunk,
            metadata={"source": "nutrition_db", "dish": dish_key},
        ))
        ids.append(f"db_{dish_key}")

    # LangChain add_documents handles embedding internally
    vs.add_documents(documents=docs, ids=ids)

    _db_populated = True
    count = col.count()
    log.info(f"ChromaDB populated — {count} total docs ✅")
    return count


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT UPSERTS  (menu + user log)
# ══════════════════════════════════════════════════════════════════════════════
def upsert_menu(menu: dict) -> None:
    """
    Upsert the mess menu into ChromaDB.
    Called once during one-time menu setup (or on menu update).
    Hash-checked — skips if menu hasn't changed.
    """
    global _last_menu_hash

    raw          = json.dumps(menu, sort_keys=True, default=str)
    current_hash = hashlib.md5(raw.encode()).hexdigest()
    if current_hash == _last_menu_hash:
        log.debug("Menu unchanged — skipping upsert.")
        return

    vs   = _get_vectorstore()
    docs = _menu_to_documents(menu)
    if not docs:
        log.warning("upsert_menu: no valid dishes found in menu dict.")
        return

    ids = [f"menu_{d.metadata['meal_id']}_{d.metadata['dish']}" for d in docs]

    # Delete old menu docs first to avoid stale entries
    try:
        vs._collection.delete(where={"source": "menu"})
    except Exception:
        pass

    vs.add_documents(documents=docs, ids=ids)
    _last_menu_hash = current_hash
    log.info(f"Menu upserted — {len(docs)} dish-meal entries indexed.")


def upsert_user_log(user_log: dict, user_id: str) -> None:
    """
    Upsert a user's 7-day meal log into ChromaDB.
    Hash-checked per user — skips if log hasn't changed.
    """
    global _last_log_hashes

    raw          = json.dumps(user_log, sort_keys=True, default=str)
    current_hash = hashlib.md5(raw.encode()).hexdigest()
    if _last_log_hashes.get(user_id) == current_hash:
        log.debug(f"Log for user '{user_id}' unchanged — skipping upsert.")
        return

    vs   = _get_vectorstore()
    docs = _log_to_documents(user_log, user_id)
    if not docs:
        return

    ids = [
        f"log_{user_id}_{d.metadata['date']}_{d.metadata['meal_id']}_{i}"
        for i, d in enumerate(docs)
    ]

    # Delete old log entries for this user
    try:
        vs._collection.delete(where={"source": "user_log", "user_id": user_id})
    except Exception:
        pass

    vs.add_documents(documents=docs, ids=ids)
    _last_log_hashes[user_id] = current_hash
    log.info(f"Log for user '{user_id}' upserted — {len(docs)} entries indexed.")


# ══════════════════════════════════════════════════════════════════════════════
# LCEL RAG CHAIN
# ══════════════════════════════════════════════════════════════════════════════
def _format_docs(docs) -> str:
    """Format retrieved LangChain Documents into a single context string."""
    texts = [d.page_content for d in docs]
    context = "\n".join(texts)
    if len(context) > MAX_CONTEXT:
        context = context[:MAX_CONTEXT] + "…"
    return context


def build_rag_chain():
    """
    Build and return the full LCEL RAG chain (singleton).

    Chain:
        { question, user_goal } 
            → retriever (parallel with passthrough)
            → prompt
            → LLM
            → StrOutputParser

    Demonstrates:
        - LangChain Retrievers     (video 13)
        - LCEL RunnableParallel    (video 8-9)
        - ChatPromptTemplate       (video 4)
        - ChatHuggingFace LLM      (video 3)
        - StrOutputParser          (video 6)
    """
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_community.llms import HuggingFaceHub

    # Ensure DB is populated before building chain
    populate_nutrition_db()

    retriever = _get_retriever()

    # ── Prompt ────────────────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are NutriAI, a nutrition assistant for Indian college students "
            "eating at a mess cafeteria. Answer using ONLY the nutrition context "
            "provided. Be specific, practical, and concise (3-5 sentences). "
            "If the context doesn't cover the question, say so honestly. "
            "User's goal: {user_goal}."
        )),
        ("human", (
            "NUTRITION CONTEXT:\n{context}\n\n"
            "QUESTION: {question}"
        )),
    ])

    # ── LLM ───────────────────────────────────────────────────────────────────
    hf_key = os.getenv("HF_API_KEY")
    if not hf_key:
        raise RuntimeError("HF_API_KEY not set in .env")

    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=hf_key,
        model_kwargs={"temperature": 0.3, "max_new_tokens": 300},
    )

    # ── LCEL Chain ────────────────────────────────────────────────────────────
    # RunnableParallel fetches context AND passes question/goal through simultaneously
    setup = RunnableParallel(
        context  = retriever | _format_docs,
        question = RunnablePassthrough(),
        user_goal= RunnablePassthrough(),
    )

    _rag_chain = setup | prompt | llm | StrOutputParser()

    log.info("LCEL RAG chain built ✅")
    return _rag_chain


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════
def retrieve_context(question: str, k: int = TOP_K) -> list[str]:
    """
    Retrieve top-k relevant chunks for a question.
    Used by chatbot.py to inject context into conversation.
    """
    retriever = _get_retriever()
    docs      = retriever.invoke(question)
    return [d.page_content for d in docs[:k]]


def answer_question(
    question  : str,
    user_goal : str  = "maintain",
    user_id   : Optional[str]  = None,
    today_menu: Optional[dict] = None,
    user_log  : Optional[dict] = None,
) -> dict:
    """
    Main RAG entry point. Called by chatbot and API endpoints.

    Args:
        question   : User's nutrition question
        user_goal  : "lose" | "gain" | "maintain"
        user_id    : For scoping log retrieval
        today_menu : Mess menu dict (upserted if provided)
        user_log   : User's 7-day log dict (upserted if provided)

    Returns:
        { "answer": str, "sources": [str], "question": str }
    """
    goal_labels = {
        "lose"    : "weight loss — calorie deficit, high protein",
        "gain"    : "muscle gain — calorie surplus, high protein",
        "maintain": "maintenance — balanced macros",
    }
    goal_str = goal_labels.get(user_goal, "balanced nutrition")

    # Upsert dynamic context if provided
    if today_menu:
        try:
            upsert_menu(today_menu)
        except Exception as e:
            log.warning(f"Menu upsert failed (non-fatal): {e}")

    if user_log and user_id:
        try:
            upsert_user_log(user_log, user_id)
        except Exception as e:
            log.warning(f"Log upsert failed (non-fatal): {e}")

    # Retrieve relevant chunks for sources list
    sources = retrieve_context(question)

    # Run LCEL chain
    try:
        chain  = build_rag_chain()
        answer = chain.invoke({
            "question" : question,
            "user_goal": goal_str,
        })
    except Exception as e:
        log.error(f"RAG chain failed: {e}")
        # Graceful fallback — return raw retrieved chunks
        if sources:
            answer = "Here's what I found:\n\n" + "\n\n".join(sources[:2])
        else:
            answer = "Sorry, I couldn't find relevant nutrition data for your question."

    return {
        "answer"  : answer,
        "sources" : sources,
        "question": question,
    }