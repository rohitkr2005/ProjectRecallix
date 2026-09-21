# Project Recallix — Developer Guide 🛠️

Welcome to the **Project Recallix Developer Guide**! This guide covers setting up your local development environment, running test suites, executing benchmarks, adding new capabilities, and troubleshooting common issues.

---

## 1. Local Environment Setup

### 1. Prerequisites
- **Python**: 3.10 or higher
- **Git**: For version control
- **Ollama (Optional)**: For local LLM inference ([Install Ollama](https://ollama.ai/))

### 2. Clone & Virtual Environment

```powershell
# Clone the repository
git clone https://github.com/rohitkr2005/ProjectRecallix.git
cd ProjectRecallix

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Database Initialization

```powershell
python main.py
```
This generates the SQLite database schema at `data/recallix.db`.

---

## 2. Local LLM Setup (Ollama)

Recallix defaults to using [Ollama](https://ollama.ai/) for local, private LLM generation.

1. Start the Ollama server:
   ```bash
   ollama serve
   ```
2. Pull the default model:
   ```bash
   ollama pull qwen2.5:3b
   # or
   ollama pull llama3:latest
   ```

> [!TIP]
> **Extractive Fallback**: If Ollama is not installed or running, Recallix automatically activates its built-in **extractive fallback** mechanism. All assistant operations, memory saving, and grounded question answering will continue to work without throwing errors.

---

## 3. Running Tests

Recallix maintains a comprehensive test suite of **261 unit, integration, and regression tests**.

### Run Full Test Suite
```powershell
python -m pytest tests/ -v
```

### Run Module-Specific Test Suites

| Target Area | Test Command |
| :--- | :--- |
| **Production Hardening (11.1–11.6)** | `pytest tests/test_production_hardening.py -v` |
| **Advanced Memory (10.1–10.8)** | `pytest tests/test_advanced_memory_intelligence.py -v` |
| **Assistant Pipeline (Phase 9)** | `pytest tests/test_assistant_intelligence.py -v` |
| **LLM Reasoning & Grounding (Phase 8)** | `pytest tests/test_llm_reasoning.py -v` |
| **Memory Lifecycle & Conflicts (Phase 7)** | `pytest tests/test_memory_lifecycle.py -v` |
| **Retrieval Ranking & Signals (Phase 6)** | `pytest tests/test_retrieval_ranking.py -v` |
| **Explainability & Metadata** | `pytest tests/test_retrieval_explainability.py -v` |
| **IR Benchmark & Regression** | `pytest tests/test_retrieval_benchmark.py -v` |

---

## 4. Running the Retrieval Benchmark

To evaluate retrieval quality and check against regression thresholds:

```powershell
python -m app.retrieval.retrieval_benchmark_runner
```

This outputs precision, recall, hit rate, MRR, NDCG, and per-query logs across the 16 benchmark cases.

---

## 5. Working with Memories Programmatically

### Storing Memories
```python
from app.assistant.assistant_engine import AssistantEngine

assistant = AssistantEngine()

# Add a fact via natural language
res = assistant.respond("I am studying Machine Learning at Mumbai University.")
print(res["extracted_memories"])

# Add facts directly via MemoryStore
memory, status, meta = assistant.memory_store.save_memory_with_semantics(
    subject="User",
    relation="knows",
    value="TypeScript",
    category="SKILL",
    importance=8,
)
print(f"Memory ID: {memory.id}, Status: {status}")
```

### Querying Memories
```python
# Query with grounded response
res = assistant.respond("What am I studying?")
print("Answer:", res["response"])
print("Grounding Score:", res["grounding_details"]["grounding_score"])
print("Latency:", res.get("performance_metrics", {}))
```

---

## 6. Extending Recallix

### Adding a New Retrieval Signal

To introduce a custom ranking signal (e.g. user-group affinity) in `app/retrieval/advanced_retrieval.py`:
1. Implement the scoring helper method:
   ```python
   def _calculate_custom_score(self, memory, query_context) -> float:
       # Return normalized float between 0.0 and 1.0
       return 1.0 if memory.category == query_context else 0.0
   ```
2. Include the new score in `_calculate_score_components`:
   ```python
   components["custom_score"] = self._calculate_custom_score(memory, query_context)
   ```
3. Update the weighted average in `_calculate_final_score`:
   ```python
   weighted = (
       0.30 * components["semantic"]
       + 0.20 * components["graph"]
       + 0.15 * components["temporal"]
       + 0.10 * components["intent"]
       + 0.10 * components["importance"]
       + 0.05 * components["recency"]
       + 0.10 * components["custom_score"]
   )
   ```

---

## 7. Troubleshooting & FAQs

### Q: Ollama connection refused (`ConnectionRefusedError`)
- **Cause**: Ollama service is not running locally.
- **Resolution**: Start Ollama with `ollama serve` or let Recallix use its automatic extractive fallback.

### Q: `MemoryValidationError`: Memory value exceeds maximum length
- **Cause**: The memory text exceeds the maximum character threshold (default 500 chars).
- **Resolution**: Set the environment variable `RECALLIX_MAX_MEMORY_LENGTH=1000` or split the statement into smaller facts.

### Q: `EmbeddingError`: Corrupted embedding size
- **Cause**: The vector byte buffer size does not equal $384 \times 4 = 1536$ bytes.
- **Resolution**: Ensure you are using the default `all-MiniLM-L6-v2` model or adjust `expected_dim` in `validate_embedding_vector`.

### Q: SQLite `database is locked` error
- **Cause**: An unclosed session holds a write lock on the SQLite file.
- **Resolution**: Always call `assistant.close()` in cleanup or use a `try...finally` block.
