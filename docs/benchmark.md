# Project Recallix — Retrieval Benchmark & Evaluation 📊

This document details the standardized benchmark dataset, information retrieval (IR) metrics, baseline performance scores, and automated regression thresholds for **Project Recallix**.

---

## 1. Overview

The Recallix Retrieval Benchmark is an automated evaluation suite designed to validate:
1. **Query Intent Understanding**: Accuracy of classifying user questions into functional domains.
2. **Retrieval Completeness & Ranking**: Measuring how effectively relevant facts are surfaced in the top-K results.
3. **Regression Protection**: Ensuring changes to vector models, graph traversals, or ranking weights never degrade retrieval quality.

The benchmark runs against an isolated, in-memory/temporary SQLite database populated with a standardized memory bank.

---

## 2. Benchmark Dataset (`app.retrieval.retrieval_benchmark`)

The dataset comprises **18 curated memory facts** and **16 test queries** spanning 6 functional domains:

### Memory Bank (18 Facts)

| Domain | Subject | Relation | Value | Base Importance |
| :--- | :--- | :--- | :--- | :---: |
| **PROJECT** | User | `works_on` | Project Recallix | 9 |
| **PROJECT** | User | `works_on` | Portfolio Website | 6 |
| **PROJECT** | User | `works_on` | Task Tracker App | 5 |
| **EDUCATION** | User | `studies` | Computer Science | 8 |
| **EDUCATION** | User | `studies_at` | Mumbai University | 8 |
| **EDUCATION** | User | `studies` | Machine Learning Course | 7 |
| **SKILL** | User | `knows` | Python | 9 |
| **SKILL** | User | `knows` | FastAPI | 8 |
| **SKILL** | User | `knows` | Docker | 7 |
| **SKILL** | User | `knows` | PostgreSQL | 7 |
| **GOAL** | User | `wants_to_learn` | Rust | 8 |
| **GOAL** | User | `wants_to_become` | Senior AI Engineer | 9 |
| **GOAL** | User | `wants_to_build` | Autonomous Agent Framework | 8 |
| **LOCATION** | User | `lives_in` | Mumbai | 8 |
| **LOCATION** | User | `previously_lived_in` | Delhi | 6 |
| **PREFERENCE** | User | `likes` | Football | 6 |
| **PREFERENCE** | User | `likes` | Sci-Fi Movies | 5 |
| **PREFERENCE** | User | `likes` | Specialty Coffee | 5 |

---

### Query Suite (16 Queries)

The query suite tests exact matches, semantic paraphrases, and ambiguous phrasing across all categories:
- `project_01`: *"What projects am I working on?"* (Expected: `PROJECT`)
- `project_02`: *"Which projects am I currently building?"* (Expected: `PROJECT`)
- `project_03`: *"Tell me about the projects I work on"* (Expected: `PROJECT`)
- `education_01`: *"What am I studying?"* (Expected: `EDUCATION`)
- `education_02`: *"What is my degree and course?"* (Expected: `EDUCATION`)
- `education_03`: *"Which subjects do I study?"* (Expected: `EDUCATION`)
- `skill_01`: *"What programming skills do I know?"* (Expected: `SKILL`)
- `skill_02`: *"Which technologies and tools do I know?"* (Expected: `SKILL`)
- `skill_03`: *"What am I good at?"* (Expected: `SKILL`)
- `goal_01`: *"What do I want to learn?"* (Expected: `GOAL`)
- `goal_02`: *"What are my future goals?"* (Expected: `GOAL`)
- `goal_03`: *"What am I planning to build?"* (Expected: `GOAL`)
- `location_01`: *"Where do I live?"* (Expected: `LOCATION`)
- `location_02`: *"What is my city?"* (Expected: `LOCATION`)
- `preference_01`: *"What sports do I like?"* (Expected: `PREFERENCE`)
- `preference_02`: *"What do I like?"* (Expected: `PREFERENCE`)

---

## 3. Evaluation Metrics

Recallix measures retrieval quality using standard Information Retrieval (IR) formulations:

### 1. Precision@K ($P@K$)
Measures the proportion of retrieved documents in the top-$K$ that are relevant:
$$P@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$

### 2. Recall@K ($R@K$)
Measures the proportion of all relevant documents that are successfully captured in the top-$K$:
$$R@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$

### 3. Hit Rate@K ($HR@K$)
Binary indicator of whether at least one relevant document was found in the top-$K$:
$$HR@K = \begin{cases} 1 & \text{if } |\text{Retrieved}_K \cap \text{Relevant}| > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 4. Mean Reciprocal Rank (MRR)
Measures the position of the first relevant document across all queries:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
Where $\text{rank}_i$ is the rank position of the first relevant memory for query $i$.

### 5. Normalized Discounted Cumulative Gain (NDCG@K)
Evaluates ranking quality, heavily penalizing relevant documents ranked lower down:
$$\text{DCG}@K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

### 6. Intent Accuracy
The fraction of test queries where the detected intent matches the ground truth:
$$\text{Accuracy}_{\text{intent}} = \frac{\sum_{i=1}^{|Q|} \mathbb{I}(\hat{y}_i = y_i)}{|Q|}$$

---

## 4. Baseline Results vs. Regression Thresholds

Recallix enforces strict regression guards in `tests/test_retrieval_benchmark.py`:

| Metric | Measured Baseline | Regression Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | **93.75%** (15/16) | $\ge 85.0\%$ | ✅ PASS |
| **Hit Rate@5** | **100.0%** (16/16) | $\ge 90.0\%$ | ✅ PASS |
| **Recall@5** | **96.88%** | $\ge 80.0\%$ | ✅ PASS |
| **MRR** | **0.9688** | $\ge 0.80$ | ✅ PASS |
| **NDCG@5** | **0.9585** | $\ge 0.85$ | ✅ PASS |
| **Precision@5** | **50.00%** | $\ge 40.0\%$ | ✅ PASS |

---

## 5. Running the Benchmark

### CLI Runner
To run the benchmark directly and inspect per-query breakdowns:
```powershell
python -m app.retrieval.retrieval_benchmark_runner
```

### Automated Pytest Suite
The benchmark is also validated automatically during pytest execution:
```powershell
python -m pytest tests/test_retrieval_benchmark.py -v
```
Any drop below the protected regression thresholds will cause the test suite to fail immediately.
