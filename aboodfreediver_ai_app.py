from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
import secrets

# Gemini (google-genai)
# pip install google-genai
try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None  # type: ignore

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Aqua – Abood Freediver Assistant", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

# -----------------------------
# In-memory storage
# -----------------------------
# NOTE: This resets on every deploy / server restart.
# If you need persistence, move to Redis/DB.
CONVERSATIONS: Dict[str, Dict[str, Any]] = {}
# session_id -> {
#   "messages": [{role, text, ts, seen}],
#   "needs_human": bool,
#   "created": str(iso),
#   "updated": str(iso),
# }

# -----------------------------
# Models
# -----------------------------
class HistoryItem(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    history: Optional[List[HistoryItem]] = None


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
    # Support both names to avoid confusion in Render.
    admin_user = os.getenv("ADMIN_USER") or os.getenv("ADMIN_USERNAME") or ""
    admin_pass = os.getenv("ADMIN_PASS") or os.getenv("ADMIN_PASSWORD") or ""

    user_ok = secrets.compare_digest(credentials.username, admin_user)
    pass_ok = secrets.compare_digest(credentials.password, admin_pass)

    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_msg(session_id: str, role: str, text: str, seen: bool) -> None:
    convo = CONVERSATIONS.setdefault(
        session_id,
        {"messages": [], "needs_human": False, "created": now_iso(), "updated": now_iso()},
    )
    convo["messages"].append({"role": role, "text": text, "ts": now_iso(), "seen": seen})
    convo["updated"] = now_iso()


def is_booking_intent(q: str) -> bool:
    q = q.lower()
    keys = [
        "book",
        "booking",
        "reserve",
        "reservation",
        "schedule",
        "tomorrow",
        "today",
        "this week",
        "available",
        "availability",
        "calendar",
        "time",
        "slot",
        "appointment",
    ]
    return any(k in q for k in keys)


# -----------------------------
# Gemini
# -----------------------------
def gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    if genai is None:
        return None
    return genai.Client(api_key=api_key)


def gemini_answer(question: str, history: Optional[List[HistoryItem]] = None) -> Tuple[str, bool]:
    """Returns (answer, needs_human)."""

    client = gemini_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"

    # Business facts (keep in sync with your website)
    business_facts = {
        "name": "Abood Freediver",
        "location": "Aqaba, Jordan (South Beach)",
        "phone": "+962 77 9617618",
        "email": "free@aboodfreediver.com",
        "contact_form": "https://www.aboodfreediver.com/form1.php",
        "prices_url": "https://www.aboodfreediver.com/Prices.php?lang=en",
        "calendar_url": "https://www.aboodfreediver.com/calender.php",
        "opening_hours": "Daily 08:00–18:00 (Aqaba local time), confirm for special dates.",
    }

    # If Gemini isn't configured, fall back to a useful deterministic response.
    if client is None:
        # Keep answers helpful even when key is missing.
        q = question.lower()
        if any(k in q for k in ["price", "cost", "how much"]):
            return f"Prices: {business_facts['prices_url']}", False
        if any(k in q for k in ["phone", "number", "whatsapp", "call"]):
            return f"You can reach us on WhatsApp/phone: {business_facts['phone']} or email: {business_facts['email']}", False
        if is_booking_intent(q):
            return (
                f"I can help, but I need to confirm the schedule. Please share your preferred date/time and course, or contact us: {business_facts['contact_form']}",
                True,
            )
        return (
            "I can help with freediving courses, safety, prices, and availability in Aqaba. "
            f"Prices: {business_facts['prices_url']} | Calendar: {business_facts['calendar_url']} | Contact: {business_facts['phone']}",
            False,
        )

    # Build conversation for Gemini
    # Keep it simple: include last N turns from history (if provided).
    parts: List[str] = []
    if history:
        # Only keep the last ~12 items to avoid huge prompts
        hist = history[-12:]
        for h in hist:
            r = (h.role or "").strip().lower()
            if r not in {"user", "assistant"}:
                continue
            label = "User" if r == "user" else "Assistant"
            parts.append(f"{label}: {h.content}")
    parts.append(f"User: {question}")

    system = f"""You are Aqua, the assistant for {business_facts['name']} (freediving courses & trips).
Be concise, friendly, and practical.

Hard facts you can use:
- Location: {business_facts['location']}
- Phone/WhatsApp: {business_facts['phone']}
- Email: {business_facts['email']}
- Opening hours: {business_facts['opening_hours']}
- Prices: {business_facts['prices_url']}
- Calendar: {business_facts['calendar_url']}

Rules:
- If the user asks about booking/availability for a specific date/time and you cannot be sure, ask 1–2 follow-up questions and set NEEDS_HUMAN = true.
- If user asks for phone/email, provide it directly (do not only send the contact form).
- Never invent prices or exact availability.
- Output MUST be valid JSON with keys:
  - answer: string
  - needs_human: boolean
"""

    prompt = system + "\n\nConversation:\n" + "\n".join(parts)

    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        raw = (resp.text or "").strip()
        # Expect JSON. If Gemini returns extra text, try to extract JSON object.
        m = None
        if raw.startswith("{") and raw.endswith("}"):
            m = raw
        else:
            import re as _re
            mobj = _re.search(r"\{.*\}", raw, _re.S)
            if mobj:
                m = mobj.group(0)

        if not m:
            return raw or "Sorry, I couldn't generate a response.", False

        import json as _json
        data = _json.loads(m)
        answer = str(data.get("answer", "")).strip() or "Sorry, I couldn't generate a response."
        needs_human = bool(data.get("needs_human", False))
        return answer, needs_human
    except Exception as e:
        # Safe fallback on any model/API error
        return (
            f"I'm having trouble reaching the AI right now. Please contact us: {business_facts['phone']} / {business_facts['email']}.",
            True,
        )


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": now_iso()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = (req.session_id or "").strip() or str(uuid.uuid4())

    add_msg(session_id, "user", req.question, seen=True)

    answer, needs_human = gemini_answer(req.question, req.history)

    add_msg(session_id, "assistant", answer, seen=True)

    if needs_human:
        CONVERSATIONS[session_id]["needs_human"] = True

    return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human)


@app.get("/chat/status")
def chat_status(session_id: str):
    convo = CONVERSATIONS.get(session_id)
    if not convo:
        return {"messages": [], "needs_human": False}

    # Mark unseen human messages as seen once fetched.
    for m in convo.get("messages", []):
        if m.get("role") == "human" and m.get("seen") is False:
            # keep it False in the payload so frontend can detect it
            # and we will flip after returning by copying (below).
            pass

    # Return a copy; flip seen for human messages in stored convo after.
    messages = []
    for m in convo.get("messages", []):
        messages.append(dict(m))

    # After creating the response copy, mark stored human messages as seen
    for m in convo.get("messages", []):
        if m.get("role") == "human" and m.get("seen") is False:
            m["seen"] = True

    return {"messages": messages, "needs_human": bool(convo.get("needs_human", False))}


@app.post("/human")
def human_reply(data: HumanReply, _: bool = Depends(admin_auth)):
    convo = CONVERSATIONS.get(data.session_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Session not found")

    add_msg(data.session_id, "human", data.message, seen=False)
    convo["needs_human"] = False

    return {"ok": True}


# -----------------------------
# Simple Admin Dashboard
# -----------------------------
@app.get("/admin", response_class=HTMLResponse)
def admin_page(_: bool = Depends(admin_auth)):
    return """<!DOCTYPE html>
<html>
<head>
  <title>Aqua – Instructor Dashboard</title>
  <style>
    body { font-family: Arial; padding: 20px; max-width: 760px }
    input, textarea, button { width: 100%; margin: 8px 0; padding: 10px; font-size: 14px }
    .hint { color: #555; font-size: 13px; margin: 6px 0 16px }
    code { background: #f3f3f3; padding: 2px 6px; border-radius: 4px }
  </style>
</head>
<body>
  <h2>Reply as Human Instructor</h2>
  <div class="hint">
    Session ID is the visitor's chat session. You can copy it from the website browser LocalStorage key
    <code>hero_session_id</code>, or from <code>/chat/status?session_id=...</code>.
  </div>

  <input id="sid" placeholder="Session ID (hero_session_id)">
  <textarea id="msg" rows="5" placeholder="Your reply to the diver"></textarea>
  <button onclick="send()">Send Reply</button>

  <pre id="out"></pre>

<script>
async function send() {
  const res = await fetch('/human', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: document.getElementById('sid').value.trim(),
      message: document.getElementById('msg').value.trim()
    })
  });
  document.getElementById('out').textContent = await res.text();
}
</script>
</body>
</html>
"""
