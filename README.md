# MDT AI Copilot

**An analytics layer and an agentic RAG copilot for 5G Minimization of Drive Tests (MDT) data from OpenAirInterface. It exports measurement reports to SQLite, detects coverage anomalies, forecasts signal strength, and lets an engineer ask questions in plain English.**

```
[UE: MDT ring buffer] ──existing──▶ [gNB: store_mdt_report(), in-memory per-UE list]
                                                │  (this project starts here)
                                                ▼
                          Export pipeline: log parser + synthetic generator
                                                ▼
                          SQLite: mdt_reports, ue_samples
                                                ▼
                          ML core: rolling z-score anomalies + RSRP forecasting
                                                ▼
                          LangGraph agent: 5 data tools + RAG over the system docs
                                                ▼
                          CLI chat  /  Streamlit dashboard  /  Next.js dashboard
```

## Background

During a two-month 5G/6G research internship at HNNOIX, I worked on a private fork of OpenAirInterface's 5G RRC stack that already implemented MDT. Phones log signal-strength samples, package them into 3GPP `MeasurementReport` messages, and the gNB (the tower) decodes and stores them in memory, per phone. The internship ended there. The data was lost on restart, alerts used fixed thresholds, and the only way to understand the system was to read the C source.

This project adds what was missing:

| Gap in the existing system | What this project adds | Why it matters |
| --- | --- | --- |
| Reports live in gNB memory and are lost on restart | SQLite export pipeline: a log parser and a synthetic generator | Persistence, analytics, ML input |
| Fixed hard-coded thresholds (3 dB, −100 dBm) | Per-UE rolling z-score anomaly detector | Adapts to each cell's own noise |
| No signal forecasting | Lag-feature GradientBoosting RSRP forecaster | Flags degradation before a threshold fires |
| You must read C source to understand the system | Agentic RAG copilot over curated system docs | Answers in plain English, grounded in real docs |

## What's inside

**Export pipeline** (`pipeline/`). `log_parser.py` parses the exact log line that `store_mdt_report()` writes in `rrc_gNB.c` and can tail a live log. `synthetic_gen.py` simulates 10 UEs for an hour at 5-second intervals (7,200 samples, seeded) and injects sustained coverage holes into about 20% of them as ground truth.

**ML core** (`ml/`). A per-UE rolling z-score detector, with IsolationForest as a comparison, and a GradientBoosting forecaster on five lag features that predicts the next RSRP reading. `evaluate.py` scores both against the injected ground truth and a predict-last-value baseline.

**Agentic RAG copilot** (`copilot/`, `rag/`). A LangGraph ReAct agent decides for each question whether it needs data, docs, both or neither. *"How many reports for UE 3?"* is a data question, *"Why a ring buffer?"* is a docs question, and *"Is cell 5's behaviour expected given the MDT trigger logic?"* needs both. The agent has five tools:

| Tool | Purpose |
| --- | --- |
| `query_reports` | MDT samples filtered by cell, UE or time window |
| `run_anomaly_scan` | Runs the detector over recent samples, optionally for one cell |
| `get_cell_summary` | Average RSRP, sample count and anomaly count for a cell |
| `suggest_threshold` | Adaptive drop and low-RSRP thresholds from each cell's noise profile |
| `retrieve_docs` | Semantic search over four curated MDT and 3GPP documents (Chroma, Cohere `embed-v4.0`) |

The guardrails: at most 8 tool calls per question, each tool's output is capped at 4,000 characters, and the system prompt requires cited numbers and forbids estimating data values.

**Dashboards** (`dashboard/`). A five-panel Streamlit app (network overview, cell health, anomaly explorer, forecasting, copilot chat), and a Next.js 14 + TypeScript dashboard served by a FastAPI backend.

## Results on the synthetic benchmark

From `python scripts/demo_phase2.py` (seed 42, 7,200 samples, 17 injected anomalous samples):

| Model | Result |
| --- | --- |
| RSRP forecaster (GradientBoosting, 5 lags) | MAE **1.04 dB** vs 1.08 dB for predict-last-value, **3.7% better** |
| Anomaly detector, default (window 20, z > 3) | Precision 0.33, recall 0.29 |
| Anomaly detector, window 50, z > 3 | Precision 0.37, recall 0.65 |

The anomaly detector is the open problem. The original target was 80% recall, and the z-score detector only reaches it at thresholds where precision falls below 5%. The next step is to combine the z-score with the `delta_rsrp_db` feature that IsolationForest already uses.

## Quick start

Needs Python 3.11+, Node.js 18+ (only for the Next.js dashboard), and a free [Cohere](https://cohere.com) API key. The key is used for the agent (`command-r-plus-08-2024`) and for embeddings (`embed-v4.0`).

```bash
pip install -r requirements.txt
cp .env.example .env            # add your COHERE_API_KEY

python scripts/demo_full.py     # every phase end to end
```

Or one step at a time:

```bash
python scripts/demo_phase1.py   # generate the synthetic dataset
python scripts/demo_phase2.py   # evaluate the anomaly detector and forecaster
python -m rag.build_index       # build the Chroma index (once)
python -m copilot.cli           # chat with the copilot
streamlit run dashboard/app.py  # Streamlit dashboard

python -m dashboard.api         # FastAPI backend for the web dashboard
cd dashboard/web && npm install && npm run dev   # http://localhost:3001
```

Run the tests with `pytest` (29 tests: log parser, anomaly detector, forecaster).

## Design decisions

- **SQLite, not a server database.** It's a single file and part of the Python standard library, so the demo runs from a cold clone on any laptop.
- **GradientBoosting, not an LSTM.** About 7,000 rows is too little for an LSTM to beat trees. GradientBoosting trains in milliseconds and its feature importances are readable.
- **A rolling z-score, not fixed thresholds.** A reading that is normal for one UE's environment is not an alarm. Each UE is judged against its own recent baseline.
- **An agent, not naive RAG.** Naive RAG retrieves for every question and wastes context on pure data questions, so the agent routes each question instead. That costs more per query, and it only pays off because the questions really are mixed.
- **Synthetic data with injected anomalies.** Real MDT data has no labels, and injected coverage holes give exact ground truth for evaluation. The log parser is ready for real gNB logs.

## Repository layout

```
pipeline/     db schema, gNB log parser, synthetic generator
ml/           anomaly detection, RSRP forecasting, evaluation
rag/          knowledge documents and the Chroma index builder
copilot/      LangGraph agent, tools, CLI
dashboard/    Streamlit app, FastAPI backend, Next.js web dashboard (web/)
scripts/      phase-by-phase and full demos
tests/        pytest suite
```
