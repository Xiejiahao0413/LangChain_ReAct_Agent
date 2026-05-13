from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or instruction.")


class ChatResponse(BaseModel):
    query: str
    answer: str


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "langchain-react-agent"
