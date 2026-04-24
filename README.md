# NutriAI 🍛  
### Multi-Agent LLM Nutrition Assistant (LangGraph, FastAPI, Docker)

> A multi-agent AI system designed for Indian college students to track nutrition, optimize diet, and get context-aware recommendations.

---

## 🚀 Key Highlights

- ⚡ **99% latency reduction** on repeat queries (26.9ms → 0.2ms, measured)  
- 🤖 **Dual-LLM ensemble system** with divergence detection (~18.8% avg disagreement observed)  
- 🧠 **Multi-agent architecture (LangGraph)** with parallel + human-in-the-loop workflows  
- ⚙️ **FastAPI backend** (25+ endpoints, async, streaming, 50+ concurrent requests)  
- 🐳 **Dockerized deployment** — one-command setup  

---

## 🧠 System Overview

5-layer pipeline:

Cache → DB → Fuzzy Match → Dual-LLM → Fallback  

- Routes majority of queries away from LLM  
- Optimized for repeated mess food usage  
- Combines caching + parallel inference for performance  

---

## 📊 Performance

| Query Type | Latency |
|-----------|--------|
| Cache (warm) | ~0.2ms |
| DB lookup | ~26.9ms |
| Dual-LLM | ~5.4s |
| Cached LLM result | <1ms |

- ~99% reduction on repeat queries  
- LLM results cached after first query  

---

## 🧪 Key Insight

Dual-model inference showed ~18.8% average disagreement across novel dishes, highlighting the need for ensemble-based responses instead of relying on a single model.

---

## 🏗️ Architecture

- FastAPI (async backend)  
- LangGraph (multi-agent workflows)  
- PostgreSQL (persistent memory)  
- ChromaDB (RAG)  
- Docker (deployment)  

---

## 🐳 Run with Docker

```bash
docker pull us0005bit/nutriai:latest
docker run -p 8000:8000 us0005bit/nutriai
```

---

## 🔗 Live API

Swagger: https://nutri-ai-0jvj.onrender.com/docs  

---

## 📌 Why this project?

Built specifically for Indian college mess systems:
- Fixed menus  
- No nutrition labels  
- Repeated meals  

---

