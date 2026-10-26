---
title: Industrial AI Diagnostics Agent
emoji: "🏭"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Industrial AI Diagnostics Agent

A production-oriented diagnostic prototype that combines a bounded LangGraph research agent with deterministic sensor telemetry analysis. It is **not** a safety-certified control system: it does not operate equipment, identify confirmed faults, predict remaining useful life, or treat anomaly scores as failure probabilities.

## What it does

- Uploads UTF-8 telemetry CSVs and validates timestamps, numeric features, row limits, duplicates, and non-finite readings.
- Runs either Isolation Forest (with robust scaling) or a robust rolling z-score baseline.
- Returns traceable timestamps, original measurements, scores, configuration, data-quality warnings, and duration.
- Builds validated Markdown or JSON diagnostic reports that label evidence as observed, calculated, hypothesis, or unknown.
- Retains the natural-language research/workspace agent with bounded graph iterations and tool timeouts.
- Confines agent file tools to `workspace/`; traversal and absolute paths are rejected.

The repository’s actual frontend is a FastAPI-served HTML console (not Gradio). Its original research input remains available, and the new telemetry controls appear beneath it.

## Architecture

`HTML console → FastAPI → telemetry/anomaly.py → validated result → diagnostics/reporting.py → downloadable report`

The optional natural-language path is separate: `HTML console → FastAPI → LangGraph → Groq/Tavily + confined workspace tools`.

## Install and run

Python 3.11 is the tested deployment target (Windows PowerShell or Linux shell):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860`. Docker uses the same command:

```bash
docker build -t industrial-diagnostics .
docker run --rm -p 7860:7860 --env-file .env industrial-diagnostics
```

For Hugging Face Docker Spaces, push this repository with its `Dockerfile`; port `7860` matches both the container and the FastAPI entry point.

## Configuration

Create a local `.env` (never commit it):

```dotenv
GROQ_API_KEY=...
TAVILY_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
MAX_GRAPH_ITERATIONS=8
TOOL_TIMEOUT_SECONDS=20
MAX_UPLOAD_BYTES=10485760
MAX_WORKSPACE_FILE_BYTES=1048576
```

The deterministic CSV workflow works without Groq or Tavily credentials. Research features require both provider keys.

## Telemetry CSV example

```csv
timestamp,temperature,pressure
2025-01-01T00:00:00Z,20.1,1.2
2025-01-01T01:00:00Z,20.0,1.3
```

Timestamps must be parseable, ascending, and unique. At least 10 rows and one numeric sensor column are required. Select features in the UI or leave the field empty to use all numeric columns.

## Testing and reproducibility

```bash
pytest -q tests
python benchmark.py
```

`benchmark.py` creates 100 deterministic synthetic cases: 80 valid telemetry files with an injected outlier and 20 invalid files missing timestamps. It writes machine-readable results to `data/benchmarks/results.json` and a short report to `data/benchmarks/report.md`. Those results describe only this synthetic/offline suite; they are not industrial accuracy claims.

For the current checked environment, pytest plugin autoload is broken by a globally installed legacy LangSmith/Pydantic-v1 package under Python 3.12. Run the project in the fresh pinned virtual environment above. As a local workaround: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests` (on Windows `set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1&& python -m pytest -q tests`).

Metrics for labelled data are provided by `telemetry.evaluation.evaluate_predictions`: precision, recall, F1, PR-AUC (only when scores and positive labels exist), confusion matrix, false positives, and false negatives. Unlabelled uploads receive no invented performance metrics. For real time-series experiments, split chronologically and fit preprocessing only on reference/training rows.

## Security and limitations

- Uploads are capped and stored per request under `data/uploads/`; generated reports use `data/reports/`.
- Agent file paths are relative to `workspace/`, UTF-8-only, size-capped, and cannot traverse outward.
- Spaces local disk is ephemeral and is not multi-user durable storage. Do not use it for sensitive or regulated production data without a suitable storage and authentication design.
- Web results are evidence, not instructions or authorization for tools. Reports distinguish observations from hypotheses but external evidence retrieval is still an optional agent task requiring human review.
