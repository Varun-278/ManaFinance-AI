from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="default")


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class HealthResponse(BaseModel):
    status: str
