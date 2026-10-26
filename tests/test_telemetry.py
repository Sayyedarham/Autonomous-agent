import numpy as np
import pandas as pd
import pytest
from telemetry.anomaly import AnalysisConfig, TelemetryValidationError, analyze_csv
from telemetry.evaluation import evaluate_predictions

def make_csv(tmp_path, rows=60):
    values = np.ones(rows) * 10
    values[-1] = 100
    path = tmp_path / "telemetry.csv"
    pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=rows, freq="h"), "temperature": values, "pressure": np.linspace(1, 2, rows)}).to_csv(path, index=False)
    return path

def test_isolation_forest_returns_traceable_anomalies(tmp_path):
    result = analyze_csv(make_csv(tmp_path), AnalysisConfig(contamination=.05))
    assert result.dataset_summary["anomaly_count"] >= 1
    assert all("timestamp" in item and "measurements" in item for item in result.anomalies)

def test_rejects_duplicate_timestamps(tmp_path):
    path = make_csv(tmp_path)
    frame = pd.read_csv(path); frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]; frame.to_csv(path, index=False)
    with pytest.raises(TelemetryValidationError, match="Duplicate"):
        analyze_csv(path)

def test_metrics_are_computed_from_labels():
    result = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 0], [.1, .9, .8, .2])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["precision"] == .5 and result["recall"] == .5
