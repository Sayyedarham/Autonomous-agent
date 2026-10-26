import pytest
from agent.files import safe_workspace_path
from diagnostics.reporting import build_report
from telemetry.anomaly import AnalysisResult

def test_workspace_cannot_escape_root():
    with pytest.raises(ValueError): safe_workspace_path("../secret.txt")
    with pytest.raises(ValueError): safe_workspace_path("C:/secret.txt")

def test_report_marks_observation_and_limitations():
    analysis = AnalysisResult({"rows": 10, "anomaly_count": 1, "anomaly_rate": .1}, ["sensor"], [], {}, [], 1.0)
    report = build_report(analysis)
    assert report.evidence[0].kind == "observed"
    assert report.uncertainty_and_limitations
