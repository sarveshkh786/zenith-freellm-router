"""
Zenith AI Gateway (router.py)
OpenAI-compatible local LLM failover gateway proxy.
Authored by Sarvesh Zenith / Dayal Infotech Services.
"""

import time
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Zenith AI Gateway Proxy",
    description="OpenAI-compatible local LLM failover gateway proxy by Sarvesh Zenith / Dayal Infotech Services",
    version="1.0.0",
)

# Backend LLM endpoints configured in priority order
LLM_BACKENDS: List[Dict[str, Any]] = [
    {"name": "Local Ollama Primary", "url": "http://localhost:11434/v1", "timeout": 5.0},
    {"name": "Local vLLM Secondary", "url": "http://localhost:8000/v1", "timeout": 5.0},
    {"name": "Local LM Studio Fallback", "url": "http://localhost:1234/v1", "timeout": 5.0},
]


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default-model")
    messages: List[ChatCompletionMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False


@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend status."""
    active_backends = []
    async with httpx.AsyncClient() as client:
        for backend in LLM_BACKENDS:
            try:
                resp = await client.get(f"{backend['url']}/models", timeout=2.0)
                if resp.status_code == 200:
                    active_backends.append(backend["name"])
            except Exception:
                pass
    return {"status": "ok", "active_backends": active_backends, "total_backends": len(LLM_BACKENDS)}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """OpenAI-compatible chat completions proxy with automatic failover."""
    payload = body.dict()
    async with httpx.AsyncClient() as client:
        for backend in LLM_BACKENDS:
            try:
                url = f"{backend['url']}/chat/completions"
                if body.stream:
                    req = client.build_request("POST", url, json=payload, timeout=backend["timeout"])
                    res = await client.send(req, stream=True)
                    if res.status_code == 200:
                        return StreamingResponse(res.aiter_raw(), media_type="text/event-stream")
                else:
                    res = await client.post(url, json=payload, timeout=backend["timeout"])
                    if res.status_code == 200:
                        return res.json()
            except Exception as e:
                # Log failover attempt and continue to next backend
                continue

    raise HTTPException(status_code=503, detail="All local LLM backend instances failed or are unreachable.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
