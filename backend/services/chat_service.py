from threading import Lock

from backend.services import financial_assistant_legacy as legacy


class ChatService:
    def __init__(self):
        self._lock = Lock()
        self._contexts = {}

    def _get_context(self, session_id: str):
        if session_id not in self._contexts:
            self._contexts[session_id] = {
                "last_symbol": None,
                "last_dates": None,
            }
        return self._contexts[session_id]

    def ask(self, session_id: str, message: str) -> str:
        with self._lock:
            context = self._get_context(session_id)
            reply, new_context = legacy.handle_query(message, context)
            self._contexts[session_id] = new_context
        return reply


chat_service = ChatService()
