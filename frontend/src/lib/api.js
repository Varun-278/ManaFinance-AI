const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function sendChat(message, sessionId = "default") {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Chat request failed");
  }

  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) {
    throw new Error("Backend unavailable");
  }
  return res.json();
}
