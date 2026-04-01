import { useEffect, useMemo, useRef, useState } from "react";
import { getHealth, sendChat } from "../lib/api";

const QUICK_PROMPTS = [
  "Compare SBIN and HDFCBANK from 2010-2020",
  "Trend analysis of RELIANCE from 2015-2022",
  "Show TCS price on 2018-05-10",
  "Compare returns of Tata Steel and HDFCBANK",
  "State Bank of India price on 2014-04-01",
];

function makeSessionId() {
  const existing = localStorage.getItem("manafinance_session_id");
  if (existing) return existing;

  const created = `session_${Date.now()}_${Math.floor(Math.random() * 10000)}`;
  localStorage.setItem("manafinance_session_id", created);
  return created;
}

export default function ChatPanel() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Welcome to ManaFinance AI. Ask about NIFTY50 prices, multi-year trends, and return comparisons.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");

  const chatBodyRef = useRef(null);

  const sessionId = useMemo(() => makeSessionId(), []);

  useEffect(() => {
    let active = true;

    async function ping() {
      try {
        await getHealth();
        if (active) setApiStatus("online");
      } catch {
        if (active) setApiStatus("offline");
      }
    }

    ping();
    const id = setInterval(ping, 10000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    if (!chatBodyRef.current) return;
    chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
  }, [messages, loading]);

  const onSend = async (overrideMessage) => {
    const message = (overrideMessage ?? input).trim();
    if (!message || loading) return;

    setLoading(true);
    setError("");
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setInput("");

    try {
      const data = await sendChat(message, sessionId);
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
    } catch {
      setError("Request failed. Check backend status and try again.");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I could not reach the backend right now. Please verify backend is running on port 8000.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  const copyText = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // no-op
    }
  };

  return (
    <div className="app-shell">
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />

      <header className="app-header glass-panel">
        <div>
          <h1>ManaFinance AI</h1>
          <p>NIFTY50 analytics with structured data, trend intelligence, and RAG-based finance assistance.</p>
        </div>
        <div className="header-status">
          <span className={`status-dot ${apiStatus}`} />
          <span className="status-text">
            API {apiStatus === "checking" ? "Checking" : apiStatus === "online" ? "Online" : "Offline"}
          </span>
        </div>
      </header>

      <main className="layout-grid">
        <aside className="left-panel glass-panel">
          <h2>Quick Start</h2>
          <p className="muted">Tap a prompt to run it instantly.</p>

          <div className="prompt-list">
            {QUICK_PROMPTS.map((prompt) => (
              <button key={prompt} className="prompt-chip" onClick={() => onSend(prompt)} disabled={loading}>
                {prompt}
              </button>
            ))}
          </div>

          <div className="session-box">
            <span className="session-label">Session</span>
            <code>{sessionId}</code>
          </div>
        </aside>

        <section className="chat-panel glass-panel">
          <div className="chat-body" ref={chatBodyRef}>
            {messages.map((m, idx) => (
              <article key={`${m.role}-${idx}`} className={`bubble-row ${m.role}`}>
                <div className="bubble">
                  <div className="bubble-head">
                    <span>{m.role === "user" ? "You" : "ManaFinance"}</span>
                    {m.role === "assistant" && (
                      <button className="copy-btn" onClick={() => copyText(m.content)}>
                        Copy
                      </button>
                    )}
                  </div>
                  <p>{m.content}</p>
                </div>
              </article>
            ))}

            {loading && (
              <article className="bubble-row assistant">
                <div className="bubble typing">
                  <div className="typing-dots">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </article>
            )}
          </div>

          <div className="composer-wrap">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask anything about NIFTY50 prices, trend windows, and return comparisons..."
              rows={2}
              disabled={loading}
            />
            <button className="send-btn" onClick={() => onSend()} disabled={loading || !input.trim()}>
              {loading ? "Analyzing..." : "Send"}
            </button>
          </div>

          {error && <p className="error-text">{error}</p>}
        </section>
      </main>
    </div>
  );
}
