"""Safe, deterministic sensor telemetry analysis. This module never controls equipment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class TelemetryValidationError(ValueError):
    """An uploaded telemetry file cannot safely be analyzed."""


@dataclass(frozen=True)
class AnalysisConfig:
    timestamp_column: str = "timestamp"
    sensor_columns: list[str] | None = None
    method: str = "isolation_forest"  # isolation_forest or robust_zscore
    contamination: float = 0.05
    rolling_window: int = 20
    zscore_threshold: float = 3.5
    min_rows: int = 10
    max_rows: int = 100_000
    random_state: int = 42


@dataclass
class AnalysisResult:
    dataset_summary: dict[str, Any]
    selected_features: list[str]
    anomalies: list[dict[str, Any]]
    model_configuration: dict[str, Any]
    data_quality_warnings: list[str]
    execution_duration_ms: float
    score_note: str = "Anomaly scores rank unusual observations; they are not calibrated probabilities of equipment failure."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_and_validate(path: Path, config: AnalysisConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    if path.suffix.lower() != ".csv":
        raise TelemetryValidationError("Only .csv telemetry files are supported.")
    try:
        frame = pd.read_csv(path)
    except UnicodeDecodeError as exc:
        raise TelemetryValidationError("Telemetry CSV must be UTF-8 encoded.") from exc
    except Exception as exc:
        raise TelemetryValidationError(f"Could not read CSV: {exc}") from exc
    if len(frame) < config.min_rows:
        raise TelemetryValidationError(f"At least {config.min_rows} rows are required; received {len(frame)}.")
    if len(frame) > config.max_rows:
        raise TelemetryValidationError(f"File has {len(frame)} rows, above configured limit {config.max_rows}.")
    if config.timestamp_column not in frame.columns:
        raise TelemetryValidationError(f"Missing required timestamp column '{config.timestamp_column}'.")
    timestamps = pd.to_datetime(frame[config.timestamp_column], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise TelemetryValidationError("Timestamp column contains unparseable or missing values.")
    if not timestamps.is_monotonic_increasing:
        raise TelemetryValidationError("Timestamps must be in ascending order; sort the source data before upload.")
    if timestamps.duplicated().any():
        raise TelemetryValidationError("Duplicate timestamps are not supported; de-duplicate source data before upload.")
    frame = frame.copy()
    frame[config.timestamp_column] = timestamps
    candidates = [c for c in frame.columns if c != config.timestamp_column and pd.api.types.is_numeric_dtype(frame[c])]
    features = config.sensor_columns or candidates
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise TelemetryValidationError(f"Selected sensor columns do not exist: {', '.join(missing)}.")
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise TelemetryValidationError(f"Sensor columns must be numeric: {', '.join(non_numeric)}.")
    if not features:
        raise TelemetryValidationError("No numeric sensor columns found. Include at least one numeric sensor column.")
    values = frame[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise TelemetryValidationError("Sensor measurements contain missing, infinite, or non-numeric values; correct them before upload.")
    warnings: list[str] = []
    if len(candidates) != len(frame.columns) - 1:
        warnings.append("Non-numeric columns were excluded from sensor analysis.")
    return frame, features, warnings


def analyze_csv(path: str | Path, config: AnalysisConfig | None = None) -> AnalysisResult:
    """Analyze a validated CSV with either Isolation Forest or robust rolling z-scores."""
    started = perf_counter()
    config = config or AnalysisConfig()
    if not 0 < config.contamination < 0.5:
        raise TelemetryValidationError("contamination must be greater than 0 and less than 0.5.")
    if config.rolling_window < 3:
        raise TelemetryValidationError("rolling_window must be at least 3.")
    frame, features, warnings = _load_and_validate(Path(path), config)
    values = frame[features].to_numpy(dtype=float)
    if config.method == "isolation_forest":
        scaled = RobustScaler().fit_transform(values)
        model = IsolationForest(contamination=config.contamination, random_state=config.random_state, n_estimators=200)
        labels = model.fit_predict(scaled) == -1
        scores = -model.score_samples(scaled)
    elif config.method == "robust_zscore":
        median = pd.DataFrame(values).rolling(config.rolling_window, min_periods=max(3, config.rolling_window // 2)).median()
        mad = (pd.DataFrame(values) - median).abs().rolling(config.rolling_window, min_periods=max(3, config.rolling_window // 2)).median()
        zscores = (pd.DataFrame(values) - median).abs() / (1.4826 * mad.replace(0, np.nan))
        aggregate = zscores.max(axis=1).fillna(0.0).to_numpy()
        labels = aggregate > config.zscore_threshold
        scores = aggregate
    else:
        raise TelemetryValidationError("method must be 'isolation_forest' or 'robust_zscore'.")
    anomalies: list[dict[str, Any]] = []
    for index in np.flatnonzero(labels):
        measurement = {name: float(frame.iloc[index][name]) for name in features}
        anomalies.append({"row_index": int(index), "timestamp": frame.iloc[index][config.timestamp_column].isoformat(), "anomaly_score": float(scores[index]), "measurements": measurement})
    summary = {"rows": int(len(frame)), "timestamp_start": frame[config.timestamp_column].iloc[0].isoformat(), "timestamp_end": frame[config.timestamp_column].iloc[-1].isoformat(), "anomaly_count": len(anomalies), "anomaly_rate": len(anomalies) / len(frame)}
    return AnalysisResult(summary, features, anomalies, asdict(config), warnings, round((perf_counter() - started) * 1000, 2))
