from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

from telemetry.anomaly import AnalysisResult


class EvidenceItem(BaseModel):
    kind: Literal["observed", "calculated", "hypothesis", "unknown"]
    claim: str
    source: str


class DiagnosticReport(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    generated_at: datetime
    executive_summary: str
    dataset_overview: dict[str, Any]
    anomalies: list[dict[str, Any]]
    evidence: list[EvidenceItem]
    uncertainty_and_limitations: list[str]
    suggested_follow_up: list[str]
    model_configuration: dict[str, Any]
    sources: list[str] = Field(default_factory=list)


def build_report(result: AnalysisResult, evidence: list[dict[str, str]] | None = None) -> DiagnosticReport:
    summary = result.dataset_summary
    observations = [EvidenceItem(kind="observed", claim=f"{summary['anomaly_count']} observations were flagged across {summary['rows']} rows.", source="telemetry analysis")]
    observations.append(EvidenceItem(kind="calculated", claim=f"Anomaly rate is {summary['anomaly_rate']:.2%}.", source="telemetry analysis"))
    for item in evidence or []:
        kind = item.get("kind", "hypothesis")
        if kind not in {"observed", "calculated", "hypothesis", "unknown"}:
            kind = "hypothesis"
        observations.append(EvidenceItem(kind=kind, claim=item.get("claim", "Unspecified external evidence."), source=item.get("source", "unattributed")))
    return DiagnosticReport(generated_at=datetime.now(timezone.utc), executive_summary=f"Telemetry analysis flagged {summary['anomaly_count']} potentially unusual observations. This is a diagnostic screening result, not a confirmed equipment-fault finding.", dataset_overview=summary, anomalies=result.anomalies, evidence=observations, uncertainty_and_limitations=[result.score_note, *result.data_quality_warnings, "Potential causes require engineering review and corroborating evidence."], suggested_follow_up=["Review flagged timestamps against maintenance and operating logs.", "Inspect affected sensors for calibration or communication issues.", "Use labelled, time-aware validation before operational decisions."], model_configuration=result.model_configuration, sources=[item.source for item in observations])


def write_report(report: DiagnosticReport, destination: str | Path, fmt: Literal["json", "markdown"] = "json") -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    elif fmt == "markdown":
        lines = ["# Industrial AI Diagnostic Report", "", f"Generated: {report.generated_at.isoformat()}", "", "## Executive summary", report.executive_summary, "", "## Dataset overview", *[f"- {k}: {v}" for k, v in report.dataset_overview.items()], "", "## Evidence"]
        lines.extend(f"- **{item.kind.title()}** — {item.claim} (Source: {item.source})" for item in report.evidence)
        lines += ["", "## Uncertainty and limitations", *[f"- {value}" for value in report.uncertainty_and_limitations], "", "## Suggested follow-up", *[f"- {value}" for value in report.suggested_follow_up]]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        raise ValueError("fmt must be json or markdown")
    return path
