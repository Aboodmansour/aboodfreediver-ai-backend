from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
import secrets

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Aqua – Abood Freediver Assistant", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

# -----------------------------
# In-memory storage
# -----------------------------
CONVERSATIONS: Dict[str, Dict] = {}
# session_id -> {
#   "messages": [{role, text, seen}],
#   "needs_human": bool,
#   "created": datetime
# }

# -----------------------------
# Models
# -----------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    needs_human: bool = False


class HumanReply(BaseModel):
    session_id: str
    message: str


# -----------------------------
# Auth
# -----------------------------
def admin_auth(credentials: HTTPBasicCredentials = Depends(security)):
    user_ok = secrets.compare_digest(
        credentials.username, os.getenv("ADMIN_USER", "")
    )
    pass_ok = secrets.compare_digest(
        credentials.password, os.getenv("ADMIN_PASS", "")
    )
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# -----------------------------
# Aqua logic
# -----------------------------
def aqua_answer(question: str) -> tuple[str, bool]:
    q = question.lower()

    if any(k in q for k in ["price", "cost", "how much"]):
        return (
            "You can see current prices here:\nhttps://www.aboodfreediver.com/Prices.php?lang=en",
            False,
        )

    if any(k in q for k in ["calendar", "availability", "available", "date"]):
        return (
            "If there is no scheduled event on the calendar, we are usually available. "
            "I just need to confirm with the instructor first 🙂",
            True,
        )

    if any(k in q for k in ["book", "booking", "reserve"]):
        return (
            "I can help with booking. Let me check availability with the instructor.",
            True,
        )

    if any(k in q for k in ["hello", "hi", "hey"]):
        return (
            "Hi, I’m Aqua 🌊 Ask me about freediving courses, prices, safety, or availability in Aqaba.",
            False,
        )

    return (
        "I can help with freediving courses, safety, prices, and availability in Aqaba. "
        "If your question needs confirmation, I’ll check with the instructor.",
        False,
    )


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    answer, needs_human = aqua_answer(req.question)

    convo = CONVERSATIONS.setdefault(
        session_id,
        {
            "messages": [],
            "needs_human": False,
            "created": datetime.utcnow(),
        },
    )

    convo["messages"].append(
        {"role": "user", "text": req.question, "seen": True}
    )

    convo["messages"].append(
        {"role": "assistant", "text": answer, "seen": True}
    )

    if needs_human:
        convo["needs_human"] = True

    return ChatResponse(
        answer=answer,
        session_id=session_id,
        needs_human=needs_human,
    )


@app.get("/chat/status")
def chat_status(session_id: str):
    convo = CONVERSATIONS.get(session_id)
    if not convo:
        return {"messages": []}

    return {
        "messages": convo["messages"],
        "needs_human": convo["needs_human"],
    }


@app.post("/human")
def human_reply(data: HumanReply, _: bool = Depends(admin_auth)):
    convo = CONVERSATIONS.get(data.session_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Session not found")

    convo["messages"].append(
        {"role": "human", "text": data.message, "seen": False}
    )
    convo["needs_human"] = False

    return {"ok": True}


# -----------------------------
# Simple Admin Dashboard
# -----------------------------
@app.get("/admin", response_class=HTMLResponse)
def admin_page(_: bool = Depends(admin_auth)):
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Aqua – Instructor Dashboard</title>
  <style>
    body { font-family: Arial; padding: 20px; max-width: 600px }
    input, textarea, button { width: 100%; margin: 8px 0; padding: 8px }
  </style>
</head>
<body>
  <h2>Reply as Human Instructor</h2>

  <input id="sid" placeholder="Session ID">
  <textarea id="msg" rows="4" placeholder="Your reply to the diver"></textarea>
  <button onclick="send()">Send Reply</button>

  <pre id="out"></pre>

<script>
async function send() {
  const res = await fetch('/human', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: document.getElementById('sid').value,
      message: document.getElementById('msg').value
    })
  });
  document.getElementById('out').textContent = await res.text();
}
</script>
</body>
</html>
"""
