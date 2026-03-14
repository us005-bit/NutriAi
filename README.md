# NutriAI 🍛
### Agentic Nutrition Assistant for Indian College Students

> A multi-agent AI system that helps college students eating at a mess cafeteria track nutrition, close protein/calorie gaps, and get context-aware dietary advice — built with LangGraph, LangChain, FastAPI, and PostgreSQL.

---

## Live API
**Swagger UI:** `https://nutriai-api.up.railway.app/docs`
> Try every endpoint live — upload food photos, ask nutrition questions, run the onboarding agent.

---

## What NutriAI Does

Most nutrition apps assume you're cooking your own food or ordering from a restaurant. NutriAI is built for the **Indian college mess reality** — a fixed weekly menu, no calorie labels, shared meals, and tight budgets.

1. **One-time setup** — Scan your mess menu board with OCR → dishes auto-identified with nutrition data
2. **Weekly planning** — Select what you'll eat each day → AI finds protein/calorie gaps → recommends specific foods to fill them
3. **Daily logging** — Log meals from the menu, type custom foods, or scan a photo → see calories and protein consumed vs remaining in real time
4. **Daily chatbot** — Ask anything: *"I skipped lunch, what should I eat for dinner?"* — chatbot knows your profile, today's intake, and your history
5. **Weekly review** — Every Sunday, get a personalised summary with insights and one actionable tip

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI (main.py)                     │
│                     25 REST endpoints                        │
└──────────┬──────────┬──────────┬──────────┬─────────────────┘
           │          │          │          │
    ┌──────▼──┐ ┌─────▼───┐ ┌───▼────┐ ┌───▼──────────────┐
    │ agent.py│ │chatbot  │ │rag.py  │ │classifier.py     │
    │         │ │   .py   │ │        │ │+ nutrition.py    │
    │3 LangGraph│ │LangGraph│ │LCEL    │ │+ ocr.py         │
    │ agents  │ │chatbot  │ │chain   │ │                  │
    └──────┬──┘ └─────┬───┘ └───┬────┘ └──────────────────┘
           │          │         │
    ┌──────▼──────────▼─────────▼──────────────────────────┐
    │                   database.py                         │
    │          Async PostgreSQL (Neon)                      │
    │    users │ mess_menu │ meal_logs │ LG checkpoints     │
    └──────────────────────────────────────────────────────┘
```

---

## AI/ML Stack — The Flagship Parts

### 1. 5-Layer Nutrition Lookup Pipeline (`nutrition.py`)

Every dish lookup goes through 5 layers in order:

```
Cache (instant) → Exact DB match → Fuzzy match → Dual-LLM parallel → Gemini fallback
```

**Layer 4 is the differentiator** — two HuggingFace models (Qwen 72B + Mixtral 8x7B) run simultaneously in threads. Both results are **averaged** for accuracy. This is ensemble inference on nutrition data — reduces hallucination vs trusting a single model.

```python
# Two models run in parallel, results averaged
with ThreadPoolExecutor(max_workers=2) as executor:
    f_primary   = executor.submit(call_qwen_72b,   dish_name)
    f_secondary = executor.submit(call_mixtral_8x7b, dish_name)
    r1, r2 = f_primary.result(), f_secondary.result()

averaged = {f: round((r1[f] + r2[f]) / 2, 1) for f in ["calories", "protein", "carbs", "fats"]}
```

### 2. Multi-Agent LangGraph System (`agent.py`)

Three independent LangGraph StateGraphs:

#### Onboarding Agent
```
parse → [all fields?] → ask → (user replies) → parse → ...
                      → compute → save_to_postgresql → END
```
- Collects 14 profile fields conversationally
- Conditional skip logic (skips gym questions if user doesn't gym)
- Computes BMR → TDEE → personalised targets using Mifflin-St Jeor + ISSN 2023 protein guidelines
- Saves directly to PostgreSQL on completion

#### Gap Analysis Agent ← New, flagship
```
compute_daily_totals (parallel, 7 days)
    → find_gaps
        → generate_recommendations (LLM per gap day)
            → await_human  ← HITL interrupt
                → [more?] → loop / END
```
- Processes all 7 days **in parallel** using `ThreadPoolExecutor`
- Finds days where calorie or protein gap exceeds 10% of target
- LLM generates specific Indian food recommendations per gap day
- **Human-in-the-loop**: graph pauses, user confirms or skips each recommendation
- Rule-based fallback if LLM unavailable

#### Weekly Review Agent (subgraph)
```
fetch_from_postgresql → rule_based_insights → llm_narrative → END
```

### 3. Context-Aware Persistent Chatbot (`chatbot.py`)

Three memory layers working simultaneously:

| Layer | What it stores | How |
|---|---|---|
| Short-term | Current conversation messages | LangGraph `add_messages` state |
| Long-term | User profile + history across sessions | PostgreSQL `AsyncPostgresSaver` |
| Semantic | Relevant nutrition facts per query | ChromaDB RAG retrieval |

**Tool-calling loop** — the chatbot decides when to call tools:
```
load_context → generate → [tool_calls?]
                              YES → tools → update_totals → generate (loop)
                              NO  → END
```

Available tools:
- `get_nutrition` — 5-layer nutrition lookup for any dish
- `get_daily_totals` — today's intake vs targets from PostgreSQL
- `get_weekly_summary` — 7-day averages and consistency

**Streaming** — tokens stream to frontend in real time via `astream_events`. Tool call notifications also streamed: `[Looking up nutrition info...]`

### 4. RAG Pipeline (`rag.py`)

Built with LangChain LCEL:
```python
chain = RunnableParallel(
    context  = retriever | format_docs,   # ChromaDB similarity search
    question = RunnablePassthrough(),
    user_goal= RunnablePassthrough(),
) | ChatPromptTemplate | HuggingFaceHub | StrOutputParser
```

Indexed sources:
- 500+ Indian mess dishes from `nutrition_db.py`
- User's mess menu (upserted after one-time OCR setup)
- User's 7-day meal log (upserted per query)

### 5. Food Image Classifier (`classifier.py`)

- Architecture: **EfficientNet-B2** via `timm`
- Training: Fine-tuned on Indian food dataset + custom mess food photos
- Inference: Top-3 predictions with confidence scores
- Uncertain predictions (< 40% confidence) flagged for manual confirmation

### 6. OCR Menu Scanner (`ocr.py`)

- Primary: **Google Cloud Vision API** — handles bad lighting, chalk boards, handwriting
- Fallback: Tesseract (if API key not set)
- Natural serving units: dishes get human-friendly descriptions (`"1 cup"`, `"2 pieces"`) instead of raw grams
- Fuzzy matching maps OCR output to `nutrition_db` keys

---

## LangGraph Concepts Demonstrated

This project covers almost every concept from the LangGraph curriculum:

| Concept | Where |
|---|---|
| StateGraph, nodes, edges | All 3 agents + chatbot |
| Sequential workflow | Onboarding, weekly review |
| Parallel workflow | Gap analysis (7 days simultaneously) |
| Conditional workflow | Onboarding skip logic, chatbot tool router |
| Iterative workflow | Onboarding Q&A loop, chatbot tool loop |
| Human in the loop (HITL) | Gap analysis recommendation confirmation |
| PostgreSQL persistence | Chatbot conversation history |
| Short-term memory | Chatbot message state |
| Long-term memory | User profile across sessions |
| Streaming | `/chat/stream` endpoint |
| Tools in LangGraph | `get_nutrition`, `get_daily_totals`, `get_weekly_summary` |
| Subgraphs | Weekly review as subgraph |
| LangSmith observability | Auto-enabled via env vars |

---

## Project Structure

```
nutriai/
├── main.py              # FastAPI — 25 endpoints
├── agent.py             # 3 LangGraph agents
├── chatbot.py           # LangGraph chatbot, 3-layer memory, streaming
├── rag.py               # LCEL RAG chain, ChromaDB
├── database.py          # Async PostgreSQL (Neon), all DB operations
├── ocr.py               # Google Vision API + Tesseract fallback
├── nutrition.py         # 5-layer nutrition lookup, dual-LLM consensus
├── nutrition_db.py      # 500+ Indian mess dishes with macros
├── model/
│   └── classifier.py    # EfficientNet-B2 food classifier
├── requirements.txt
├── .env.example
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/docs` | Swagger UI — try all endpoints |
| GET | `/health` | Model + device info |
| POST | `/predict` | Image → dish + nutrition |
| POST | `/nutrition` | Dish name → nutrition (5-layer) |
| POST | `/suggest` | Over target → food swap suggestion |
| POST | `/ocr/scan` | Menu photo → matched dishes |
| POST | `/ocr/save-menu` | Save confirmed menu to DB |
| POST | `/onboarding/start` | Start onboarding conversation |
| POST | `/onboarding/reply` | Send reply → next question or plan |
| GET | `/user/{user_id}` | Fetch profile + targets |
| POST | `/log/meal` | Log a meal item |
| GET | `/log/totals/{user_id}` | Today's macros vs targets |
| GET | `/log/history/{user_id}` | Date range log history |
| POST | `/gap/start` | Start gap analysis |
| POST | `/gap/confirm` | Confirm/skip recommendation (HITL) |
| GET | `/review/{user_id}` | 7-day review |
| POST | `/chat` | Chat message → full response |
| POST | `/chat/stream` | Chat message → streaming response |
| GET | `/chat/history/{user_id}` | Conversation history |
| POST | `/rag/ask` | Direct RAG nutrition Q&A |

---

## Setup & Run

### Prerequisites
- Python 3.11+
- PostgreSQL (or Neon free tier)
- Tesseract binary (optional — only for OCR fallback)
  - Ubuntu: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

### Installation

```bash
# Clone repo
git clone https://github.com/yourusername/nutriai.git
cd nutriai

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the server
uvicorn main:app --reload --port 8000

# Open Swagger UI
# http://localhost:8000/docs
```

### Required API Keys

| Key | Where to get | Required? |
|---|---|---|
| `DATABASE_URL` | [neon.tech](https://neon.tech) | ✅ Yes |
| `HF_API_KEY` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | ✅ Yes |
| `GOOGLE_VISION_API_KEY` | [Google Cloud Console](https://console.cloud.google.com) | Recommended |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Optional |
| `LANGSMITH_API_KEY` | [smith.langchain.com](https://smith.langchain.com) | Optional |

---

## Deployment (Railway)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Set all environment variables in Railway dashboard under Variables.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Agent Framework | LangGraph 0.2 |
| LLM Orchestration | LangChain 0.3 |
| LLMs | Qwen 72B, Mixtral 8x7B, Llama 3.3 70B (HuggingFace) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store | ChromaDB |
| Database | PostgreSQL on Neon (async psycopg3) |
| Computer Vision | EfficientNet-B2 (timm + PyTorch) |
| OCR | Google Cloud Vision API + Tesseract fallback |
| Fuzzy Matching | rapidfuzz |
| Observability | LangSmith |

---

## Team

| Role | Contribution |
|---|---|
| AI/ML + Backend | Nutrition pipeline, LangGraph agents, chatbot, RAG, OCR, FastAPI |
| Frontend | HTML/CSS/JS — 5-page mobile app *(separate team)* |

---

## Notes

- The EfficientNet-B2 model weights (`efficientnet_b2_best.pth`) are not included in this repo due to file size. Contact the repo owner for access.
- `nutrition_cache.json` is auto-generated on first run — do not commit it.
- ChromaDB (`.chroma_db/`) is auto-populated on startup — do not commit it.

---

*Built as a flagship AI/ML project demonstrating production-grade agentic AI architecture.*
