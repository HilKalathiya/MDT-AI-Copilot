# MDT AI Copilot

┌──────────────────────────────────────────────────────────────────────┐
│ AI analytics and agentic RAG copilot for OpenAirInterface 5G MDT   │
└──────────────────────────────────────────────────────────────────────┘

> **AI analytics and agentic RAG copilot for OpenAirInterface 5G MDT**

Built an AI analytics and agentic RAG copilot (LangGraph, tool + retrieval routing) on OpenAirInterface's 5G MDT pipeline — added coverage-anomaly detection, RSRP forecasting, and a natural-language network assistant grounded in the project's own docs and codebase, closing a data-export gap in the existing system.

---

## 📖 Table of Contents

1. [What This Project Is](#what-this-project-is)
2. [The Problem We Solved](#the-problem-we-solved)
3. [System Architecture](#system-architecture)
4. [What We Built — Phase by Phase](#what-we-built--phase-by-phase)
   - [Phase 0 — Scaffold](#phase-0--scaffold)
   - [Phase 1 — Export Pipeline](#phase-1--export-pipeline)
   - [Phase 2 — ML Core](#phase-2--ml-core)
   - [Phase 3 — Agentic RAG Copilot](#phase-3--agentic-rag-copilot)
   - [Phase 4 — Dashboards & Polish](#phase-4--dashboards--polish)
5. [Repository Structure](#repository-structure)
6. [Quick Start](#quick-start)
7. [Key Design Decisions](#key-design-decisions)
8. [Technologies Used](#technologies-used)
9. [Testing](#testing)
10. [Background](#background)

---

## What This Project Is

This is a **portfolio project** built on top of a real 5G research internship codebase. During the internship (HNNOIX), we worked on a private fork ("Duranta") of OpenAirInterface's 5G RRC stack that already had an **MDT (Minimization of Drive Tests)** feature implemented:

- Phones (UEs) log signal-strength samples every few seconds
- They package those samples into standard 3GPP `MeasurementReport` messages
- The gNB (tower) receives, decodes, and stores them in per-UE memory

The internship ended there. **This project adds everything the existing system was missing:** a way to persist and export that data, machine learning to understand it, and an agentic AI copilot to reason about it in natural language.

---

## The Problem We Solved

| Gap in Existing System | What We Added | Why It Matters |
| --- | --- | --- |
| Reports live in gNB memory only — lost on restart | **SQLite export pipeline** (log parser + synthetic generator) | Persistent storage, analytics, ML input |
| Fixed hard-coded thresholds (3 dB, −100 dBm) | **Per-UE rolling z-score anomaly detector** | Adapts to each cell's actual noise profile |
| No signal forecasting | **Lag-feature GradientBoosting RSRP predictor** | Flag degradation *before* a threshold fires |
| Must read C source to understand system behavior | **Agentic RAG copilot** with conceptual retrieval | Natural-language interface grounded in real docs |

---

## System Architecture

```
[UE: MDT ring buffer]  ──existing──▶  [gNB: store_mdt_report(), in-memory per-UE list]
                                              │
                               (this project starts here)
                                              ▼
                               [Module 1: Export Pipeline]
                                log parser  +  synthetic generator
                                              ▼
                               [SQLite: mdt_reports, ue_samples]
                                              ▼
                               [Module 2: ML Core]
                                anomaly detection + RSRP forecasting
                                              ▼
                               [Module 3: Agentic RAG Copilot]
                                LangGraph agent — structured tools + RAG
                                              ▼
                    [CLI Chat]  /  [Streamlit Dashboard]  /  [Next.js Dashboard]
```

---

## What We Built — Phase by Phase

### Phase 0 — Scaffold

**Goal:** Stand up the repo structure and database schema from scratch.

**What we created:**

- Full repository layout with all module directories
- `pipeline/db.py` — initializes the SQLite database with two tables:
  - `mdt_reports` — gNB-side data (as received/decoded at the tower)
  - `ue_samples` — UE-side data (richer, mirrors the C struct `nr_mdt_sample_t`)
- Both tables have proper indexes on `ue_rrc_id`, `logged_at`, and `serving_cell_id`
- The `ue_samples` table includes an `is_injected_anomaly` ground-truth column — this is what makes the ML evaluation possible (you don't get free ground truth from real data)
- `requirements.txt` with all Python dependencies
- `.env` / `.env.example` for API key configuration
- `setup.bat` and `setup.sh` for one-command environment setup

**Definition of done:** `python -m pipeline.db --init` creates a database matching the exact schema from the project spec. ✅

---

### Phase 1 — Export Pipeline

**Goal:** Get MDT data out of gNB memory and into SQLite.

Two independent data sources were built:

#### Log Parser (`pipeline/log_parser.py`)

Parses actual gNB log output lines into the `mdt_reports` table using a regex pattern matching the exact format logged by `store_mdt_report()` in `rrc_gNB.c`:

```
[MDT][gNB UE 3] stored report #12 measId=1 serving_cell=1 serving_RSRP=-89 dBm neighbor_cell=2 neighbor_RSRP=-97 dBm
```

- Handles both neighbor-present and no-neighbor line variants
- Supports `tail -f` style live ingestion (polls a growing log file)
- Includes `parse_log_line()` for single-line parsing and `tail_and_ingest()` for streaming

#### Synthetic Generator (`pipeline/synthetic_gen.py`)

Generates a fully realistic synthetic dataset without needing a running 5G simulator:

- Creates 10 simulated UEs with random-walk RSRP signals
- Simulates 1 hour of samples at 5-second intervals (~720 samples/UE, ~7,200 total rows)
- **Deliberately injects coverage holes** into ~20% of UEs (the anomaly ground truth)
- Each hole is a sustained RSRP drop of 30–90 seconds at a random point mid-recording
- All randomness is seeded (`seed=42`) so every run is reproducible
- Correctly assigns trigger reasons (`periodic`, `meas_update`, `rsrp_drop`, `low_rsrp`) based on the same thresholds the real gNB uses

#### Demo Script (`scripts/demo_phase1.py`)

Runs `init_db()` + `generate_synthetic_dataset()` and prints row counts as a sanity check.

**Definition of done:** ≥10 UEs, ≥1 hour of samples, ≥2 anomalous UEs, unit tests pass. ✅

---

### Phase 2 — ML Core

**Goal:** Make the system smarter than fixed-threshold rules.

Two ML models were built and evaluated:

#### Anomaly Detection (`ml/anomaly.py`)

**Primary: Rolling Z-Score Detector**

- Computes a per-UE rolling mean and standard deviation over `rsrp_dbm`
- Flags any sample where `abs(z_score) > threshold` (default: 3.0)
- Window size: 20 samples (100 seconds of real-time at 5-second intervals)
- Key advantage over fixed thresholds: adapts to each UE's own recent baseline — a signal that's "normal for that UE" won't false-positive

**Stretch: IsolationForest Detector**

- Uses `sklearn.ensemble.IsolationForest` over two features: `rsrp_dbm` and `delta_rsrp_db`
- Provides an interesting comparison point: "I tried two approaches, here's the tradeoff"
- Contamination parameter set to 5%

#### RSRP Forecasting (`ml/forecast.py`)

- **Model:** `GradientBoostingRegressor` from scikit-learn
- **Features:** 5 lag features (`lag_1` through `lag_5`) — the previous 5 RSRP readings per UE
- **Target:** The *next* RSRP value (shift by -1)
- **Evaluation:** MAE on a held-out time slice vs. a naive "predict last value" baseline
- Beats the naive baseline consistently
- Trains in milliseconds on ~7,000 rows

Why GBR over LSTM: The dataset is too small (~7k rows) for an LSTM to outperform tree-based methods. GBR is interpretable (feature importances), trains instantly, and is easier to explain in an interview.

#### Evaluation (`ml/evaluate.py`)

- Computes precision/recall/F1 for the anomaly detector against `is_injected_anomaly` ground truth
- Computes model MAE vs. naive MAE for the forecaster
- Anomaly recall: **≥80%** on synthetic set

**Definition of done:** Anomaly recall ≥80%, forecaster beats naive MAE baseline. ✅

---

### Phase 3 — Agentic RAG Copilot

**Goal:** Give an engineer a natural-language interface grounded in real data and real docs.

This is the most architecturally interesting part of the project. Instead of a fixed pipeline ("always retrieve then answer"), we built a **LangGraph ReAct agent** that decides per-question whether it needs structured data, unstructured docs, both, or neither.

#### Why agentic RAG (not naive RAG)?

The question space is genuinely heterogeneous:

- *"How many reports do we have for UE 3?"* — pure data question, SQL only
- *"Why does the code use a ring buffer?"* — pure conceptual question, docs only
- *"Cell 5 looks unhealthy — is that expected given MDT trigger logic?"* — needs both

Naive RAG always retrieves, wasting tokens on data questions. An agent that chooses per-question covers all three correctly.

#### RAG Pipeline (`rag/build_index.py`)

Built a Chroma vector store (local, no external service) from four curated knowledge documents:

- `mdt_system_overview.txt` — how the MDT system works end-to-end
- `mdt_data_structures.txt` — the C structs (`nr_mdt_sample_t`, `nr_mdt_report_t`) and their fields
- `architecture_design.txt` — design decisions and system architecture rationale
- `3gpp_mdt_background.txt` — 3GPP standard context and MDT specification background

Documents are chunked at 500 tokens with 50-token overlap and embedded using **Cohere `embed-v4.0`**.

#### Structured Tools (`copilot/tools.py`)

Five LangChain tools that give the agent direct access to live data:

| Tool | Purpose |
| --- | --- |
| `query_reports` | Query MDT samples with optional filters (cell, UE, time window) |
| `run_anomaly_scan` | Run the anomaly detector over recent samples, optionally scoped to one cell |
| `get_cell_summary` | Avg RSRP, sample count, anomaly count for a cell |
| `suggest_threshold` | Suggest adaptive drop/low-RSRP thresholds based on each cell's noise profile |
| `retrieve_docs` | Semantic search over the 4 knowledge documents (returns top-k chunks) |

Each tool truncates its output to 4,000 characters to prevent context overflow.

#### The Agent (`copilot/agent.py`)

Built with `langgraph.prebuilt.create_react_agent` (LangGraph v1.0, stable since Oct 2025):

- **LLM:** Cohere `command-r-plus-08-2024` (native tool-calling support, free tier)
- **ReAct loop:** Reason → select tool → observe result → repeat until answer is ready
- **Guardrails:** Max 8 tool calls per question (`MAX_TOOL_CALLS` in `.env`), configurable
- **System prompt:** Instructs the agent to cite specific numbers, declare whether data is synthetic, and never estimate data values

#### CLI Interface (`copilot/cli.py`)

A plain chat loop: type a question, see the answer. Shows which tools were called and what they returned.

**Definition of done:** Agent correctly answers 2 structured questions, 2 conceptual questions grounded in the knowledge docs (not the model's general knowledge), and 1 hybrid question. ✅

---

### Phase 4 — Dashboards & Polish

**Goal:** Make the system demoable from a cold clone in under 2 minutes.

#### Streamlit Dashboard (`dashboard/app.py`)

A 5-panel interactive dashboard:

1. **Network Overview** — RSRP heatmap, UE count, anomaly rate
2. **Cell Health** — per-cell stats (avg RSRP, sample count, anomaly count)
3. **Anomaly Explorer** — interactive time-series with anomaly flags highlighted
4. **RSRP Forecasting** — actual vs. predicted RSRP with configurable horizon
5. **AI Copilot Chat** — embedded chat panel powered by the LangGraph agent

#### FastAPI Backend (`dashboard/api.py`)

REST API that exposes all ML and data functionality for the Next.js frontend:

- `GET /api/cell-summary` — summary stats for all cells
- `GET /api/anomalies` — recent anomalies
- `GET /api/forecast` — forecasted RSRP values
- `POST /api/chat` — streams agent responses

#### Next.js Web Dashboard (`dashboard/web/`)

A premium dark-mode web interface built with Next.js 14 + TypeScript:

- Real-time data fetching from the FastAPI backend
- Interactive charts (Plotly.js)
- Embedded AI copilot chat panel
- Runs at `http://localhost:3001`

#### Demo Scripts

| Script | Purpose |
| --- | --- |
| `scripts/demo_phase1.py` | Generate synthetic data, print row counts |
| `scripts/demo_phase2.py` | Run ML evaluation, print anomaly recall + MAE |
| `scripts/demo_full.py` | End-to-end demo — all phases in sequence |

**Definition of done:** Cold clone + `pip install` + `demo_full.py` runs in under 2 minutes. ✅

---

## Repository Structure

```
mdt-ai-copilot/
├── PROJECT_GUIDE.md          # Full specification and architecture deep-dive
├── README.md                 # This file
├── requirements.txt          # All Python dependencies
├── .env.example              # API key template
├── setup.bat / setup.sh      # One-command environment setup
│
├── data/
│   ├── mdt.db                # SQLite database (gitignored — generated)
│   └── chroma/               # Chroma vector store (gitignored — generated)
│
├── pipeline/
│   ├── db.py                 # init_db(), SQLite schema
│   ├── log_parser.py         # gNB log parser (regex + tail-f ingestion)
│   └── synthetic_gen.py      # Synthetic UE dataset generator with injected anomalies
│
├── ml/
│   ├── anomaly.py            # Rolling z-score + IsolationForest anomaly detection
│   ├── forecast.py           # Lag-feature GradientBoosting RSRP forecaster
│   └── evaluate.py           # Precision/recall and MAE evaluation
│
├── rag/
│   ├── build_index.py        # Chroma vector store builder (Cohere embed-v4.0)
│   └── docs_source/          # 4 knowledge documents for RAG indexing
│       ├── mdt_system_overview.txt
│       ├── mdt_data_structures.txt
│       ├── architecture_design.txt
│       └── 3gpp_mdt_background.txt
│
├── copilot/
│   ├── tools.py              # 4 structured tools + retrieve_docs (5 total)
│   ├── agent.py              # LangGraph create_react_agent (Cohere LLM)
│   └── cli.py                # CLI chat loop
│
├── dashboard/
│   ├── app.py                # Streamlit dashboard (5 panels)
│   ├── api.py                # FastAPI backend for Next.js
│   └── web/                  # Next.js 14 + TypeScript premium web dashboard
│       ├── src/app/page.tsx
│       └── src/app/globals.css
│
├── scripts/
│   ├── demo_phase1.py        # Data generation demo
│   ├── demo_phase2.py        # ML evaluation demo
│   └── demo_full.py          # Full end-to-end demo (<2 min)
│
└── tests/
    ├── test_log_parser.py    # 5 hand-written test cases (incl. no-neighbor variant)
    ├── test_anomaly.py       # Column presence, spike detection, FPR, threshold sensitivity
    └── test_forecast.py      # Lag features, training, MAE, predict_next()
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for Next.js dashboard only)
- **Cohere API key** — for embeddings (`embed-v4.0`) and LLM (`command-r-plus-08-2024`) — both free at [cohere.com](https://cohere.com)

### 1. Install dependencies

```bash
git clone <this-repo>
cd mdt-ai-copilot
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your COHERE_API_KEY
```

### 3. Run the full demo

```bash
python scripts/demo_full.py
```

Or step by step:

```bash
# Phase 1: Generate data
python scripts/demo_phase1.py

# Phase 2: Run ML evaluation
python scripts/demo_phase2.py

# Phase 3: Build RAG index (run once)
python -m rag.build_index

# Phase 3: Start CLI copilot
python -m copilot.cli

# Phase 4: Launch Streamlit dashboard
streamlit run dashboard/app.py
```

### 4. Next.js web dashboard

```bash
# Terminal 1 — FastAPI backend
python -m dashboard.api

# Terminal 2 — Next.js frontend
cd dashboard/web
npm install
npm run dev
# → http://localhost:3001
```

---

## Key Design Decisions

### Why SQLite?

Zero operational overhead, single file, Python stdlib — runs on any laptop with no infrastructure setup. Perfect for a portfolio demo that must work from a cold clone.

### Why GradientBoosting over LSTM?

The dataset is ~7,000 rows (10 UEs × 720 samples). LSTMs need significantly more data to outperform tree-based methods at this scale. GBR with lag features is interpretable (feature importances available), trains in milliseconds, and beats the naive baseline cleanly.

### Why rolling z-score over fixed thresholds?

The existing gNB uses fixed 3 dB / −100 dBm thresholds. These are suboptimal for cells with high natural variance. A per-UE rolling z-score adapts to each UE's own recent baseline — it catches anomalies that are anomalous *for that UE*, not just absolutely low.

### Why agentic RAG and not naive RAG?

The question set is heterogeneous: pure data questions (SQL, no retrieval), pure conceptual questions (retrieval, no SQL), and hybrid questions. Naive RAG always retrieves regardless — wasting tokens for data questions. An agent that decides per-question covers all three correctly.

**Important caveat:** agentic RAG is not automatically "better" — it costs more per query and is overkill for a simple single-corpus FAQ bot where naive RAG is cheaper and just as accurate. It earns its place here specifically because the tool surface is genuinely heterogeneous.

### Why `create_react_agent` and not `AgentExecutor`?

`AgentExecutor` is the pre-LangGraph approach and doesn't integrate cleanly with the current LangChain ecosystem. `create_react_agent` is the stable standard as of LangGraph v1.0 (October 2025) and correctly implements the ReAct (Reason → Act → Observe) loop.

### Why Cohere for both embeddings and LLM?

Cohere's `embed-v4.0` model is one of the best embedding models available and has a free tier. `command-r-plus-08-2024` natively supports tool calling and is also free for developers — meaning the entire project runs with a single free API account.

### Why synthetic data?

The real OAI/Duranta system requires a full 5G simulator to produce live MDT log output. Synthetic data with seeded randomness (`seed=42`) means the demo runs from any laptop, every run produces the same results, and — critically — we get free ground-truth anomaly labels (`is_injected_anomaly`) for evaluating the ML models. You can't get that for free from real data.

---

## Technologies Used

| Category | Technology |
| --- | --- |
| Language | Python 3.11+ |
| Database | SQLite (stdlib) |
| ML | scikit-learn (GradientBoostingRegressor, IsolationForest) |
| Data | pandas, numpy |
| LLM Agent | LangGraph `create_react_agent` (v1.0) |
| LLM | Cohere `command-r-plus-08-2024` |
| Embeddings | Cohere `embed-v4.0` |
| Vector Store | Chroma (local, embedded) |
| RAG Framework | LangChain + langchain-chroma |
| Dashboard (Python) | Streamlit + Plotly |
| API Backend | FastAPI + Uvicorn |
| Web Dashboard | Next.js 14 + TypeScript |
| Testing | pytest + pytest-cov |
| Environment | python-dotenv |

---

## Testing

```bash
pytest tests/ -v
```

**Test coverage:**

| File | What it tests |
| --- | --- |
| `tests/test_log_parser.py` | 5 hand-written gNB log lines — including a no-neighbor variant, a non-MDT line that should be ignored, and edge-case RSRP values |
| `tests/test_anomaly.py` | Column presence, spike detection (injected −20 dBm drop), false positive rate on clean signal, threshold sensitivity |
| `tests/test_forecast.py` | Lag feature construction, model training completes, MAE evaluation, `predict_next()` with exact input |

---

## Agent Guardrails

Two guardrails prevent the agent from misbehaving:

1. **Max tool calls per question:** 8 (configurable via `MAX_TOOL_CALLS` in `.env`) — prevents a confused agent from looping forever
2. **Tool output truncation:** 4,000 characters max per tool result — prevents context window overflow from large query results

---

## Phase Completion Summary

| Phase | Description | Status |
| --- | --- | --- |
| **Phase 0** | Scaffold — repo structure + SQLite schema | ✅ Done |
| **Phase 1** | Export pipeline — log parser + synthetic generator + unit tests | ✅ Done |
| **Phase 2** | ML core — anomaly detection (recall ≥80%) + RSRP forecasting (beats naive baseline) | ✅ Done |
| **Phase 3** | Agentic RAG copilot — LangGraph agent answers structured, conceptual, and hybrid questions | ✅ Done |
| **Phase 4** | Polish — Streamlit dashboard, FastAPI backend, Next.js web dashboard, full demo script | ✅ Done |

---

## Background

This project extends a 2-month 5G/6G research internship (HNNOIX). The internship codebase is a private fork ("Duranta") of OpenAirInterface's 5G RRC stack with an MDT feature already implemented: phones log signal-strength samples and report them to the tower, which stores them in memory per phone.

The internship ended at the data-in-memory stage. This project adds what the existing system is missing: **data export, ML analytics, and an agentic RAG copilot** — the complete analytics layer on top of a real 5G measurement pipeline.

See `PROJECT_GUIDE.md` for the full specification, architecture deep-dive, and interview preparation notes (including the key concepts to explain for each component).

---

## One-Line Summary (for resume/portfolio)

> *Built an AI analytics and agentic RAG copilot (LangGraph, tool + retrieval routing) on OpenAirInterface's 5G MDT pipeline — added coverage-anomaly detection (rolling z-score, recall ≥80%), RSRP forecasting (GradientBoosting, beats naive baseline), and a natural-language network assistant grounded in the project's own docs and codebase, closing the data-export gap in the existing system.*

---

## Architecture

```
[UE: MDT ring buffer]  ──existing──▶  [gNB: store_mdt_report(), in-memory per-UE list]
                                              │
                               (this project starts here)
                                              ▼
                               [Module 1: Export Pipeline]
                                log parser  +  synthetic generator
                                              ▼
                               [SQLite: mdt_reports, ue_samples]
                                              ▼
                               [Module 2: ML Core]
                                anomaly detection + RSRP forecasting
                                              ▼
                               [Module 3: Agentic RAG Copilot]
                                LangGraph agent — structured tools + RAG
                                              ▼
                    [CLI Chat]  /  [Streamlit Dashboard]  /  [Next.js Dashboard]
```

## Gap → Solution

| Gap in existing system | What this project adds | Why it matters |
| --- | --- | --- |
| Reports live in gNB memory only — lost on process restart | SQLite export pipeline (log parser + synthetic generator) | Persistence, analytics, ML input |
| Fixed hard-coded thresholds (3 dB, −100 dBm) | Per-UE rolling z-score anomaly detector | Adapts to each cell's actual noise profile |
| No signal forecasting | Lag-feature GradientBoosting RSRP predictor | Flag degradation *before* a threshold fires |
| Must read C source to understand system behavior | Agentic RAG copilot with conceptual retrieval | Natural-language interface grounded in real docs |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+ (for Next.js dashboard)
- OpenRouter API key (for the agent — `openai/gpt-oss-20b:free`, free tier)
- Cohere API key (for embeddings — `embed-v4.0`)

### 1. Install dependencies

```bash
git clone <this-repo>
cd mdt-ai-copilot
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY and COHERE_API_KEY
```

### 3. Run the full demo

```bash
python scripts/demo_full.py
```

Or step by step:

```bash
# Phase 1: Generate data
python scripts/demo_phase1.py

# Phase 2: Run ML evaluation
python scripts/demo_phase2.py

# Phase 3: Build RAG index (once)
python -m rag.build_index

# Phase 3: Start CLI copilot
python -m copilot.cli

# Phase 4: Launch Streamlit dashboard
streamlit run dashboard/app.py
```

### 4. Next.js web dashboard

```bash
# Start the FastAPI backend (required for the web dashboard)
python -m dashboard.api

# In a separate terminal, start the Next.js frontend
cd dashboard/web
npm install
npm run dev
# → http://localhost:3001
```

---

## Repository Structure

```
mdt-ai-copilot/
├── PROJECT_GUIDE.md          # Full specification
├── README.md
├── requirements.txt
├── .env.example              # API key template
├── data/
│   ├── mdt.db                # gitignored — generated by demo scripts
│   └── chroma/               # vector store — gitignored
├── pipeline/
│   ├── db.py                 # init_db(), SQLite schema (Section 5.1)
│   ├── log_parser.py         # gNB log parser (Section 5.2)
│   └── synthetic_gen.py      # Synthetic UE data generator (Section 5.3)
├── ml/
│   ├── anomaly.py            # Z-score + IsolationForest anomaly detection
│   ├── forecast.py           # Lag-feature GradientBoosting RSRP forecaster
│   └── evaluate.py           # Precision/recall and MAE evaluation
├── rag/
│   ├── build_index.py        # Chroma vector store builder
│   └── docs_source/          # Documentation for RAG indexing
│       ├── mdt_system_overview.txt
│       ├── mdt_data_structures.txt
│       ├── architecture_design.txt
│       └── 3gpp_mdt_background.txt
├── copilot/
│   ├── tools.py              # 4 structured tools + retrieve_docs
│   ├── agent.py              # LangGraph create_react_agent
│   └── cli.py                # CLI chat loop
├── dashboard/
│   ├── app.py                # Streamlit dashboard (5 panels)
│   ├── api.py                # FastAPI backend for Next.js
│   └── web/                  # Next.js premium web dashboard
│       ├── src/app/page.tsx
│       └── src/app/globals.css
├── scripts/
│   ├── demo_phase1.py
│   ├── demo_phase2.py
│   └── demo_full.py          # Full end-to-end demo (< 2 min)
└── tests/
    ├── test_log_parser.py    # 5 hand-written test cases
    ├── test_anomaly.py
    └── test_forecast.py
```

---

## Key Design Decisions

### Why SQLite?

Zero operational overhead, single file, Python stdlib — perfect for a portfolio project that must "run from a cold clone" without any infrastructure setup.

### Why GradientBoosting over LSTM?

The dataset is ~7,000 rows (10 UEs × 720 samples). LSTMs need significantly more data to outperform tree-based methods here. GBR with lag features is interpretable (feature importances available) and trains in milliseconds.

### Why rolling z-score over fixed thresholds?

The existing gNB uses fixed 3 dB / −100 dBm thresholds. These are suboptimal for cells with high natural variance. A per-UE rolling z-score adapts to each UE's own recent baseline — it catches anomalies that are anomalous *for that UE*, not just absolutely.

### Why agentic RAG and not naive RAG?

The question set is heterogeneous: pure data questions (SQL, no retrieval), pure conceptual questions (retrieval, no SQL), and hybrid questions. Naive RAG always retrieves regardless — wasting tokens for data questions and still missing them for conceptual ones. An agent that decides per-question covers all three correctly.

See `PROJECT_GUIDE.md` Section 7.1 for the full explanation, including when *not* to use agentic RAG.

### Why LangGraph's `create_react_agent` and not `AgentExecutor`?

`AgentExecutor` is the pre-LangGraph approach — it doesn't integrate cleanly with the current ecosystem. `create_react_agent` is the stable standard as of LangGraph v1.0 (Oct 2025) and correctly implements the ReAct (reason → act → observe) loop.

---

## Testing

```bash
pytest tests/ -v
```

Tests cover:

- Log parser: 5 hand-written test cases (including no-neighbor line variant)
- Anomaly detector: column presence, spike detection, false positive rate, threshold sensitivity
- Forecaster: lag feature construction, model training, MAE evaluation, predict_next()

---

## Agent Guardrails

Per Section 7.4 of `PROJECT_GUIDE.md`:

1. **Max tool calls per question**: 8 (configurable via `MAX_TOOL_CALLS` in `.env`)
2. **Tool output truncation**: 4,000 characters max per result (prevents context overflow)

---

## Phase Definitions of Done

| Phase | Criteria | Status |
| --- | --- | --- |
| Phase 0 | `python -m pipeline.db --init` creates correct schema | ✅ |
| Phase 1 | ≥10 UEs, ≥1h samples, ≥2 anomalous UEs, unit tests pass | ✅ |
| Phase 2 | Anomaly recall ≥80%, forecaster beats naive MAE baseline | ✅ |
| Phase 3 | Agent answers 2 structured + 2 conceptual + 1 hybrid correctly | ✅ |
| Phase 4 | Cold clone + `pip install` + `demo_full.py` runs in < 2min | ✅ |

---

## Background

This is a portfolio project extending a 2-month 5G/6G research internship (HNNOIX). The internship codebase is a private fork ("Duranta") of OpenAirInterface's 5G RRC stack with an MDT feature already implemented. This project adds what the existing system is missing: data export, ML analytics, and an agentic RAG copilot.

See `PROJECT_GUIDE.md` for the full specification, architecture deep-dive, and interview preparation notes.

# MDT-AI-Copilot
