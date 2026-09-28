from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from app.metrics import AI_FALLBACKS
from app.provider import providers


app = FastAPI(title="AI Infra Lab")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):

    try:
        result = providers["qwen"].chat(request.message)

    except Exception:
        AI_FALLBACKS.labels(
            from_provider="qwen",
            to_provider="deepseek",
        ).inc()

        result = providers["deepseek"].chat(request.message)

    return ChatResponse(response=result)