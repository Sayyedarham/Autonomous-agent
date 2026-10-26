"""HTTP API for bounded agent execution and deterministic telemetry diagnostics."""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from diagnostics.reporting import build_report, write_report
from telemetry.anomaly import AnalysisConfig, AnalysisResult, TelemetryValidationError, analyze_csv

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR, REPORT_DIR = BASE_DIR / "data" / "uploads", BASE_DIR / "data" / "reports"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))
for directory in (UPLOAD_DIR, REPORT_DIR): directory.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("industrial_diagnostics")
app = FastAPI(title="Industrial AI Diagnostics Agent", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.mount("/static", StaticFiles(directory=BASE_DIR / "ui"), name="static")

@app.middleware("http")
async def request_logging(request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("request_id=%s method=%s path=%s failed", request_id, request.method, request.url.path)
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s method=%s path=%s status=%s", request_id, request.method, request.url.path, response.status_code)
    return response

class GoalRequest(BaseModel): goal: str = Field(min_length=1, max_length=10_000)
class ReportRequest(BaseModel):
    analysis: dict
    format: str = Field(default="markdown", pattern="^(markdown|json)$")

def _agent():
    """Delay optional provider imports so deterministic diagnostics can still run offline."""
    try:
        from agent.graph import agent
        return agent
    except Exception as exc:
        raise HTTPException(503, detail=f"Agent integration is unavailable: {type(exc).__name__}: {exc}") from exc

@app.get("/")
def root(): return FileResponse(BASE_DIR / "ui" / "index.html")

@app.get("/health")
def health(): return {"status": "ok", "service": "industrial-ai-diagnostics", "agent_available": bool(os.getenv("GROQ_API_KEY"))}

@app.post("/run")
def run(req: GoalRequest):
    request_id = str(uuid4())
    result = _agent().invoke({"messages": [{"role": "user", "content": req.goal}], "iterations": 0}, config={"recursion_limit": int(os.getenv("MAX_GRAPH_ITERATIONS", "8")) * 2})
    if result.get("terminal_error"): raise HTTPException(502, detail={"request_id": request_id, "error": result["terminal_error"]})
    return {"request_id": request_id, "response": result["messages"][-1].content}

async def _store_upload(upload: UploadFile, request_id: str) -> Path:
    if not upload.filename or Path(upload.filename).suffix.lower() != ".csv": raise HTTPException(415, "Upload a .csv telemetry file.")
    payload = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES: raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES} byte limit.")
    destination = UPLOAD_DIR / f"{request_id}.csv"
    destination.write_bytes(payload)
    return destination

@app.post("/telemetry/analyze")
async def analyze_telemetry(file: UploadFile = File(...), method: str = Form("isolation_forest"), contamination: float = Form(0.05), rolling_window: int = Form(20), sensor_columns: str = Form("")):
    request_id = str(uuid4()); path = await _store_upload(file, request_id)
    columns = [value.strip() for value in sensor_columns.split(",") if value.strip()] or None
    try:
        result = analyze_csv(path, AnalysisConfig(method=method, contamination=contamination, rolling_window=rolling_window, sensor_columns=columns))
    except TelemetryValidationError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(422, detail={"request_id": request_id, "error": str(exc)}) from exc
    return {"request_id": request_id, "analysis": result.to_dict()}

@app.post("/reports")
def create_report(payload: ReportRequest):
    request_id = str(uuid4())
    try:
        report = build_report(AnalysisResult(**payload.analysis))
        target = REPORT_DIR / f"diagnostic-{request_id}.{ 'md' if payload.format == 'markdown' else 'json'}"
        write_report(report, target, "markdown" if payload.format == "markdown" else "json")
    except Exception as exc:
        raise HTTPException(422, detail={"request_id": request_id, "error": f"Invalid analysis for report: {exc}"}) from exc
    return {"request_id": request_id, "report": report.model_dump(mode="json"), "download_url": f"/reports/{target.name}"}

@app.get("/reports/{filename}")
def download_report(filename: str):
    if Path(filename).name != filename or not filename.startswith("diagnostic-"): raise HTTPException(404, "Report not found.")
    target = REPORT_DIR / filename
    if not target.is_file(): raise HTTPException(404, "Report not found.")
    return FileResponse(target, filename=filename)

@app.post("/stream")
async def stream(req: GoalRequest):
    async def events():
        try:
            result = _agent().invoke({"messages": [{"role": "user", "content": req.goal}], "iterations": 0}, config={"recursion_limit": int(os.getenv("MAX_GRAPH_ITERATIONS", "8")) * 2})
            if result.get("terminal_error"): yield f"data: {json.dumps({'type': 'error', 'message': result['terminal_error']})}\n\n"
            else: yield f"data: {json.dumps({'type': 'final', 'content': result['messages'][-1].content})}\n\n"
        except Exception as exc: yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
