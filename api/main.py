from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import asyncio

from agent.graph import agent

app = FastAPI(title="Autonomous AI Agent", version="1.0.0")

# Allow all origins for local dev + HF Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="ui"), name="static")

@app.get("/")
def root():
    return FileResponse("ui/index.html")

graph = agent


class GoalRequest(BaseModel):
    goal: str


@app.get("/health")
def health():
    return {"status": "ok", "agent": "autonomous-agent-v1"}


@app.post("/run")
def run(req: GoalRequest):
    """Blocking endpoint — waits for full agent response."""
    result = graph.invoke({"messages": [{"role": "user", "content": req.goal}]})
    final = result["messages"][-1].content
    return {"response": final}


@app.post("/stream")
async def stream(req: GoalRequest):
    """SSE streaming endpoint — emits tokens + step events as they happen."""

    async def event_generator():
        try:
            for chunk in graph.stream(
                {"messages": [{"role": "user", "content": req.goal}]},
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        # Tool call steps
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                event = {
                                    "type": "tool_call",
                                    "tool": tc["name"],
                                    "args": tc["args"],
                                }
                                yield f"data: {json.dumps(event)}\n\n"
                                await asyncio.sleep(0)

                        # Tool results
                        elif hasattr(msg, "name") and msg.name:
                            event = {
                                "type": "tool_result",
                                "tool": msg.name,
                                "content": str(msg.content)[:500],  # truncate long results
                            }
                            yield f"data: {json.dumps(event)}\n\n"
                            await asyncio.sleep(0)

                        # Final AI message (no tool calls)
                        elif hasattr(msg, "content") and msg.content:
                            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                                event = {
                                    "type": "final",
                                    "content": msg.content,
                                }
                                yield f"data: {json.dumps(event)}\n\n"
                                await asyncio.sleep(0)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # important for nginx proxies (HF Spaces)
        },
    )
