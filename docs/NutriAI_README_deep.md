# NutriAI — Deep Dive 🧠

## ⚡ System Goal

Design a scalable, low-latency nutrition assistant for Indian college mess environments where:
- Food repeats weekly  
- No structured nutrition data exists  
- Real-time recommendations are needed  

---

## 🧠 Core Design Decision

Avoid expensive LLM calls unless necessary by using a layered pipeline.

---

## ⚡ 5-Layer Pipeline

Cache → DB → Fuzzy → Dual-LLM → Fallback  

- Cache handles repeat queries  
- Database handles known dishes  
- LLM used only for novel inputs  

---

## 📊 Benchmarking

### Cache Benchmark
- Cold: ~26.9ms  
- Warm: ~0.2ms  
- ~99% improvement  

### LLM Benchmark
- ~5.4s average latency  
- Cached after first query  

---

## 🤖 Dual-LLM Ensemble

Two independent models:
- Qwen  
- Llama  

Observed:
- ~18.8% average disagreement  
- Indicates variability in single-model outputs  

---

## ⚙️ Concurrency

- Tested ~50+ concurrent requests  
- Enabled via FastAPI async architecture  

---

## 🐳 Deployment

- Docker multi-stage builds  
- Docker Hub image available  
- Compatible with Render / Railway  

---

## 🚀 Possible Improvements

- Redis caching  
- Better evaluation benchmarks  
- Streaming optimization  

---

