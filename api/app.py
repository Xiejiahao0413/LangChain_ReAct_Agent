from __future__ import annotations

import json
from functools import lru_cache
from typing import Iterator

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from agent.react_agent import ReactAgent
from api.schemas import ChatRequest, ChatResponse, HealthResponse


app = FastAPI(
    title="LangChain ReAct Agent API",
    description="FastAPI service for robot-vacuum ReAct Agent chat and SSE streaming.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_agent() -> ReactAgent:
    return ReactAgent()


def format_sse(event: str, payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def stream_agent_response(query: str, agent: ReactAgent) -> Iterator[str]:
    yield format_sse("start", {"query": query})
    try:
        for chunk in agent.execute_stream(query):
            if chunk:
                yield format_sse("message", {"delta": chunk})
        yield format_sse("done", {"query": query})
    except Exception as exc:
        yield format_sse("error", {"message": str(exc)})


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    agent: ReactAgent = Depends(get_agent),
) -> ChatResponse:
    try:
        answer = "".join(agent.execute_stream(request.query)).strip()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(query=request.query, answer=answer)


@app.get("/api/chat/stream")
def chat_stream_get(
    query: str = Query(..., min_length=1),
    agent: ReactAgent = Depends(get_agent),
) -> StreamingResponse:
    return StreamingResponse(
        stream_agent_response(query, agent),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/stream")
def chat_stream_post(
    request: ChatRequest,
    agent: ReactAgent = Depends(get_agent),
) -> StreamingResponse:
    return StreamingResponse(
        stream_agent_response(request.query, agent),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
