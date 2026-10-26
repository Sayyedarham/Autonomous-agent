"""Deterministic telemetry ingestion, anomaly detection, and evaluation."""

from .anomaly import AnalysisConfig, AnalysisResult, analyze_csv

__all__ = ["AnalysisConfig", "AnalysisResult", "analyze_csv"]
