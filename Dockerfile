# ─────────────────────────────────────────────────────────────────────────────
# NutriAI — Dockerfile
# Multi-stage build:
#   builder  → install Python deps + download ML models
#   runtime  → lean image with only what's needed to run
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System deps needed to compile some wheels (psycopg binary, Pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first for layer caching
COPY requirements.txt .

# Install all Python deps into a prefix we can copy to runtime stage
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="NutriAI Team"
LABEL description="NutriAI FastAPI backend — agentic nutrition assistant"

# ── System runtime libs ───────────────────────────────────────────────────────
# tesseract-ocr  : pytesseract fallback OCR
# libpq5         : psycopg binary runtime
# libglib / libsm / libxext / libxrender : OpenCV / Pillow transitive deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libpq5 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# ── App setup ─────────────────────────────────────────────────────────────────
WORKDIR /app

# Copy application source (root-level files)
COPY agent.py chatbot.py database.py main.py nutrition.py \
     nutrition_db.py ocr.py rag.py ./

# Copy pre-built nutrition cache (read-only reference data)
COPY nutrition_cache.json ./

# Copy classifier + model weights from the model/ subfolder
# classifier.py lands at /app/model/classifier.py (matches its import paths)
# efficientnet_b2_best.pth lands alongside it at /app/model/
RUN mkdir -p /app/model /app/.chroma_db
COPY model/classifier.py model/efficientnet_b2_best.pth ./model/

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN groupadd -r nutriai && useradd -r -g nutriai -d /app nutriai \
 && chown -R nutriai:nutriai /app

USER nutriai

# ── Environment defaults (override at runtime via --env-file or -e) ───────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:${PORT}/health').raise_for_status()"

EXPOSE ${PORT}

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 2"]