"""Reproducible offline benchmark for deterministic telemetry and validation tasks."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from telemetry.anomaly import AnalysisConfig, TelemetryValidationError, analyze_csv

def main() -> None:
    root = Path("data/benchmarks"); root.mkdir(parents=True, exist_ok=True)
    successes, durations, tasks = 0, [], []
    for number in range(100):
        path = root / f"case-{number}.csv"
        valid = number % 5 != 0
        frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=30, freq="h"), "temperature": np.r_[np.ones(29) * 20, 100]})
        if not valid: frame = frame.drop(columns="timestamp")
        frame.to_csv(path, index=False)
        started = perf_counter()
        try:
            result = analyze_csv(path, AnalysisConfig(contamination=.1))
            ok = valid and result.dataset_summary["anomaly_count"] >= 1
        except TelemetryValidationError:
            ok = not valid
        duration = (perf_counter() - started) * 1000
        successes += int(ok); durations.append(duration); tasks.append({"case": number, "valid_input": valid, "success": ok, "duration_ms": round(duration, 2)})
    summary = {"total_tasks": len(tasks), "successful_tasks": successes, "success_rate": successes / len(tasks), "latency_ms": {"p50": float(np.percentile(durations, 50)), "p95": float(np.percentile(durations, 95))}, "notes": "Offline deterministic validation/telemetry benchmark; no LLM, search, or claims about real industrial accuracy."}
    (root / "results.json").write_text(json.dumps({"summary": summary, "tasks": tasks}, indent=2), encoding="utf-8")
    (root / "report.md").write_text(f"# Benchmark report\n\n- Tasks: {summary['total_tasks']}\n- Success rate: {summary['success_rate']:.1%}\n- P50 latency: {summary['latency_ms']['p50']:.2f} ms\n- P95 latency: {summary['latency_ms']['p95']:.2f} ms\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
if __name__ == "__main__": main()
