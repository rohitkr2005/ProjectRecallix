# Project Recallix 🧠⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-261%20passed-brightgreen.svg)]()
[![Benchmark](https://img.shields.io/badge/retrieval%20hit--rate-100%25-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0.0-blueviolet.svg)](RELEASE_NOTES.md)

> **Project Recallix** is a high-performance, long-term memory-augmented conversational assistant that endows local and cloud LLMs with persistent, evolving, and grounded memory.

---

## 📖 Table of Contents

- [Why Recallix?](#-why-recallix)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [5-Line Quickstart](#5-line-quickstart)
- [How It Works](#-how-it-works)
- [Retrieval Benchmark & Performance](#-retrieval-benchmark--performance)
- [Repository Structure](#-repository-structure)
- [Documentation & Deep Dives](#-documentation--deep-dives)
- [Development & Testing](#-development--testing)
- [License](#-license)

---

## 💡 Why Recallix?

Traditional Large Language Models (LLMs) suffer from three fundamental limitations in conversational applications:
1. **Amnesia**: Every session starts from zero; LLMs forget user preferences, facts, and past projects.
2. **Context Limits**: Shoving chat history into the prompt window is expensive, introduces attention degradation, and quickly exhausts context tokens.
3. **Hallucination**: When asked about the user, models fabricate plausibly sounding answers instead of admitting ignorance or citing verified facts.

**Recallix solves this by acting as an external, cognitive memory cortex**:
- **Zero Hallucinations**: Enforces strict grounding verification; if a question is unsupported by memory, Recallix gracefully declines instead of making things up.
- **Dynamic Memory Evolution**: Tracks memory lifecycle—conflicting facts (e.g. moving from Delhi to Mumbai) are automatically resolved, superseded, or updated.
- **Temporal Horizons**: Distinguishes between past facts ("Previously worked at Google"), present facts ("Currently lives in Mumbai"), and future aspirations ("Plans to learn Rust").
- **Knowledge Graph Reasoning**: Connects entities and multi-hop relationships (`User` $\rightarrow$ `works_on` $\rightarrow$ `Recallix` $\rightarrow$ `uses` $\rightarrow$ `Python`).

---

## ✨ Key Features

| Capability | Description |
| :--- | :--- |
| **🧠 Memory Extraction** | Automatically parses user statements into structured subject-relation-value triplets with categorization and importance scoring. |
| **🔄 Lifecycle & Conflict Resolution** | Semantic conflict detection, non-destructive archival, supersede chains, and safe restoration. |
| **🌐 Memory Graph** | Directed knowledge graph supporting neighborhood search, multi-hop traversal, and graph proximity scoring. |
| **⏳ Temporal Awareness** | Tri-state temporal classification (`PAST`, `PRESENT`, `FUTURE`) for queries and memories, allowing coexistence of historical and current facts. |
| **🎯 Hybrid 6-Signal Retrieval** | Ranks memories combining: Semantic (35%), Graph Proximity (20%), Temporal Alignment (15%), Intent Relevance (10%), Learned Importance (10%), and Recency Decay (10%). |
| **📈 Importance Learning** | Dynamically reinforces memories that lead to grounded answers and penalizes noisy memories. |
| **🛡️ Grounded Reasoning** | Generates LLM answers strictly anchored to retrieved facts with post-generation grounding verification and extractive fallback. |
| **⚡ Production Hardened** | Centralized configuration, structured JSON logging, sub-millisecond `LatencyTracker`, input sanitization, and vector safety guards. |

---

## 🏛️ System Architecture

```
                    ┌─────────────────────────┐
                    │       User Input        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    AssistantEngine      │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │ Memory Extraction│            │ Query Intent &   │
       │ & Sanitization   │            │ Temporal Analysis│
       └─────────┬────────┘            └─────────┬────────┘
                 │                               │
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │ Lifecycle &      │            │ Hybrid 6-Signal  │
       │ Conflict Resolve │            │ Retrieval Engine │
       └─────────┬────────┘            └─────────┬────────┘
                 │                               │
                 ▼                               ▼
       ┌──────────────────┐            ┌──────────────────┐
       │   MemoryStore    │◄───────────│   MemoryGraph    │
       │  (SQLite+Vec)    │            │ (Graph Proximity)│
       └─────────┬────────┘            └─────────┬────────┘
                 │                               │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Context Builder     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       LLM Engine        │
                    │  (Ollama / Local LLM)   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Grounding Verification  │
                    │   & Feedback Learner    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Grounded Answer Output  │
                    └─────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.10 or higher
- **Local LLM Runtime (Optional for inference)**: [Ollama](https://ollama.ai/) running `qwen2.5:3b` or `llama3:latest` (extractive fallback works without Ollama).

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rohitkr2005/ProjectRecallix.git
   cd ProjectRecallix
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Initialize database**:
   ```powershell
   python main.py
   ```

5. **Run the interactive showcase demo**:
   ```powershell
   python demo.py
   ```

---

### 5-Line Quickstart

```python
from app.assistant.assistant_engine import AssistantEngine

# Initialize the assistant cortex
assistant = AssistantEngine()

# 1. State facts (automatically extracted, vectorized, and stored)
assistant.respond("I am building Project Recallix in Python.")
assistant.respond("I moved from Delhi to Mumbai.")

# 2. Ask grounded questions
result = assistant.respond("Where do I currently live?")
print(result["response"])
# -> "You currently live in Mumbai."

# 3. Ask about unsupported facts (graceful decline, zero hallucination)
result = assistant.respond("What is my favorite movie?")
print(result["response"])
# -> "I don't have that information in my memory."
```

---

## 🔍 How It Works

1. **Input Classification**: Incoming text is classified as `STATEMENT`, `QUESTION`, `BOTH`, or `NEITHER` (chit-chat).
2. **Memory Extraction & Lifecycle**: Statements extract semantic facts (`subject`, `relation`, `value`). The lifecycle manager inspects active memories for conflicts or updates and automatically archives superseded facts.
3. **Temporal Analysis**: Analyzes whether assertions or questions relate to past, present, or future horizons.
4. **Hybrid Retrieval**: Computes composite ranking scores across 6 signals (Semantic, Graph, Temporal, Intent, Learned Importance, Recency).
5. **Grounded Generation**: Feeds retrieved memories into the local LLM with strict instructions never to hallucinate.
6. **Answer Grounding Verification**: Verifies that every assertion in the response is directly supported by retrieved memories.
7. **Reinforcement Learning**: Reinforces cited memories with positive feedback, boosting their priority in future searches.

---

## 📊 Retrieval Benchmark & Performance

Recallix includes an automated retrieval evaluation benchmark tested across 18 curated memories and 16 diverse queries covering 6 intent categories (`PROJECT`, `EDUCATION`, `SKILL`, `GOAL`, `LOCATION`, `PREFERENCE`):

```
================================================================
        PROJECT RECALLIX RETRIEVAL BENCHMARK
================================================================
Intent Accuracy   : 93.75%  (15/16 correct)
Hit Rate@5        : 100.0%  (Target: >= 90.0%)
Recall@5          : 96.88%  (Target: >= 80.0%)
MRR (Mean Recip.) : 0.9688  (Target: >= 0.80)
NDCG@5            : 0.9585  (Target: >= 0.85)
Regression Check  : PASS ✅ All protected thresholds satisfied
================================================================
```

To run the benchmark locally:
```powershell
python -m app.retrieval.retrieval_benchmark_runner
```

---

## 📁 Repository Structure

```
ProjectRecallix/
├── app/
│   ├── assistant/              # High-level assistant pipeline & orchestration
│   │   └── assistant_engine.py # Unified AssistantEngine interface
│   ├── config.py               # Centralized Settings & environment variables
│   ├── database/               # SQLite schema, models, and session management
│   │   ├── database.py
│   │   └── models.py
│   ├── embeddings/             # SentenceTransformer embedding engine
│   │   └── embedding_engine.py
│   ├── llm/                    # Local LLM reasoning, grounding, and fallback
│   │   └── llm_engine.py
│   ├── memory/                 # Core memory layer
│   │   ├── forgetting.py       # Controlled forgetting & decay
│   │   ├── graph.py            # MemoryGraph knowledge relations
│   │   ├── importance_learning.py # Feedback & reinforcement learning
│   │   ├── lifecycle.py        # Update semantics & conflict resolution
│   │   ├── memory_extractor.py # Regex & heuristic statement extractor
│   │   ├── memory_store.py     # Database CRUD & vector operations
│   │   └── temporal.py         # Temporal state detection & alignment
│   ├── retrieval/              # Hybrid retrieval & evaluation
│   │   ├── advanced_retrieval.py # 6-signal hybrid retrieval engine
│   │   ├── retrieval_benchmark.py # Benchmark dataset & metrics
│   │   ├── retrieval_benchmark_runner.py # CLI benchmark runner
│   │   ├── retrieval_engine.py # Core retrieval engine
│   │   └── retrieval_evaluation.py # IR metrics (P@K, R@K, MRR, NDCG)
│   └── utils/                  # Production hardening utilities
│       ├── exceptions.py       # Domain exception hierarchy
│       ├── logging.py          # Structured JSON logging & event helpers
│       ├── metrics.py          # LatencyTracker sub-millisecond timer
│       └── validators.py       # Input sanitization & vector validation
├── docs/                       # Detailed documentation
│   ├── api.md                  # Complete public API reference
│   ├── architecture.md         # Deep dive into system architecture
│   ├── benchmark.md            # Benchmark dataset and evaluation guide
│   └── development.md          # Developer setup, testing, and contribution
├── tests/                      # 261 comprehensive unit & integration tests
│   ├── test_advanced_memory_intelligence.py
│   ├── test_assistant_intelligence.py
│   ├── test_assistant_pipeline.py
│   ├── test_llm_reasoning.py
│   ├── test_memory_lifecycle.py
│   ├── test_production_hardening.py
│   ├── test_query_understanding.py
│   ├── test_retrieval_benchmark.py
│   ├── test_retrieval_evaluation.py
│   ├── test_retrieval_explainability.py
│   └── test_retrieval_ranking.py
├── main.py                     # Database initialization entrypoint
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📚 Documentation & Deep Dives

For detailed guides, please explore our `docs/` directory:
- 🏛️ [**Architecture Guide**](docs/architecture.md): Complete layered architecture breakdown.
- 📘 [**API Reference**](docs/api.md): Detailed specifications of all classes, methods, and configurations.
- 📊 [**Benchmark Guide**](docs/benchmark.md): Metric formulations, dataset overview, and regression testing.
- 🛠️ [**Development Guide**](docs/development.md): Environment setup, writing tests, and extending Recallix.

---

## 🧪 Development & Testing

Run the full test suite (261 tests):
```powershell
python -m pytest tests/ -v
```

Run focused test modules:
```powershell
# Production Hardening
python -m pytest tests/test_production_hardening.py -v

# Advanced Memory Intelligence
python -m pytest tests/test_advanced_memory_intelligence.py -v

# LLM Reasoning & Grounding
python -m pytest tests/test_llm_reasoning.py -v
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
