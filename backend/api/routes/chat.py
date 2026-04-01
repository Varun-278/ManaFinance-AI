from fastapi import APIRouter

from backend.models.schemas import ChatRequest, ChatResponse
from backend.services.chat_service import chat_service

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(payload: ChatRequest):
    reply = chat_service.ask(payload.session_id, payload.message)
    return ChatResponse(reply=reply, session_id=payload.session_id)
