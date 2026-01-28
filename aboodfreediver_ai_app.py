from __future__ import annotations

import os
import re
import uuid
import time
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import secrets
import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Aqua – Abood Freediver Assistant", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()


# -----------------------------
# In-memory storage (Render free tier resets on deploy/sleep)
# -----------------------------
CONVERSATIONS: Dict[str, Dict[str, Any]] = {}
# session_id -> {
#   "messages": [{role, text, ts}],
#   "needs_human": bool,
#   "created": iso,
# }

# -----------------------------
# Models
# -----------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None  # optional, if your frontend sends it


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    needs_human: bool = False
    source: str = "fallback"  # "gemini" or "fallback"


class HumanReply(BaseModel):
    session_id: str
    message: str


# -----------------------------
# Admin Auth (supports both env var naming styles)
# -----------------------------
def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v)
    return default


def admin_auth(credentials: HTTPBasicCredentials = Depends(security)):
    expected_user = _env("ADMIN_USER", "ADMIN_USERNAME")
    expected_pass = _env("ADMIN_PASS", "ADMIN_PASSWORD")

    user_ok = secrets.compare_digest(credentials.username or "", expected_user or "")
    pass_ok = secrets.compare_digest(credentials.password or "", expected_pass or "")

    if not (user_ok and pass_ok):
        # Forces browser to show login popup
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# -----------------------------
# Email notify (optional)
# -----------------------------
def send_owner_email(subject: str, body: str) -> None:
    """
    Uses SMTP_* env vars:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM
    Sends to OWNER_NOTIFY_EMAIL.
    """
    owner_to = _env("OWNER_NOTIFY_EMAIL")
    if not owner_to:
        return

    smtp_host = _env("SMTP_HOST")
    smtp_port = int(_env("SMTP_PORT", default="587"))
    smtp_user = _env("SMTP_USER")
    smtp_pass = _env("SMTP_PASSWORD")
    smtp_from = _env("SMTP_FROM", default=smtp_user)

    if not (smtp_host and smtp_user and smtp_pass and smtp_from):
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = owner_to
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
    except Exception:
        # do not crash chat flow if email fails
        return


# -----------------------------
# Calendar check (basic scrape + cache)
# -----------------------------
CALENDAR_URL = _env("CALENDAR_URL", default="https://www.aboodfreediver.com/calender.php")

_calendar_cache: Dict[str, Any] = {"ts": 0.0, "events": []}


def fetch_calendar_events() -> List[str]:
    """
    Tries to extract upcoming event date lines from your public calendar page.
    If it can't find any, returns [].
    Cached for 10 minutes to avoid hammering your site.
    """
    now = time.time()
    if now - _calendar_cache["ts"] < 600 and isinstance(_calendar_cache.get("events"), list):
        return _calendar_cache["events"]

    try:
        r = requests.get(CALENDAR_URL, timeout=10)
        r.raise_for_status()
        html = r.text

        # Example seen on your page screenshot: "Sat, May 2, 2026"
        # We'll capture a few date-like strings.
        date_regex = re.compile(
            r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b",
            re.IGNORECASE,
        )
        dates = list(dict.fromkeys(date_regex.findall(html)))  # unique, preserve order
        dates = dates[:5]

        _calendar_cache["ts"] = now
        _calendar_cache["events"] = dates
        return dates
    except Exception:
        _calendar_cache["ts"] = now
        _calendar_cache["events"] = []
        return []


# -----------------------------
# Gemini (supports either new or old SDK)
# -----------------------------
def try_gemini_answer(question: str, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        return None

    model = _env("GEMINI_MODEL", default="gemini-2.5-flash")

    system = (
        "You are Aqua, the freediving assistant for Abood Freediver in Aqaba, Jordan (Red Sea).\n"
        "Rules:\n"
        "1) Be brief, clear, helpful.\n"
        "2) If the user asks about prices, link to: https://www.aboodfreediver.com/Prices.php?lang=en\n"
        "3) If the user asks about contact/booking, link to: https://www.aboodfreediver.com/form1.php\n"
        "4) If the user asks about availability/dates:\n"
        "   - If calendar has events, mention the next dates.\n"
        "   - If calendar has no events, say we are usually free BUT must confirm with the instructor.\n"
        "5) If user asks opening hours, answer with the standard hours the instructor uses (if unknown, ask them to contact).\n"
        "6) If uncertain, ask 1 short follow-up question.\n"
    )

    # Add calendar context
    dates = fetch_calendar_events()
    if dates:
        cal_context = "Upcoming dates from the calendar: " + ", ".join(dates)
    else:
        cal_context = "Calendar shows no upcoming dates (may mean mostly free; must confirm with instructor)."

    # Build message list
    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": system + "\n\n" + cal_context})

    if history:
        for m in history[-12:]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": question})

    # --- Try new SDK first: google-genai ---
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        # Convert to a single prompt for compatibility
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs])

        resp = client.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception:
        pass

    # --- Fallback old SDK: google-generativeai ---
    try:
        import google.generativeai as genai_old  # type: ignore

        genai_old.configure(api_key=api_key)
        m = genai_old.GenerativeModel(model_name=model, system_instruction=system + "\n\n" + cal_context)

        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs if m["role"] != "system"])
        r = m.generate_content(prompt)
        text = getattr(r, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception:
        return None

    return None


# -----------------------------
# Fallback logic (smarter than “stupid” answers)
# -----------------------------
def fallback_answer(question: str) -> Tuple[str, bool]:
    q = question.strip().lower()

    # prices
    if any(k in q for k in ["price", "prices", "cost", "how much", "fee"]):
        return ("Prices are here: https://www.aboodfreediver.com/Prices.php?lang=en", False)

    # booking/contact
    if any(k in q for k in ["book", "booking", "reserve", "whatsapp", "phone", "contact", "email"]):
        return ("To book or ask details: https://www.aboodfreediver.com/form1.php", True)

    # availability/calendar
    if any(k in q for k in ["availability", "available", "calendar", "date", "dates", "schedule", "time slot", "time are you free"]):
        dates = fetch_calendar_events()
        if dates:
            return (
                "Upcoming calendar dates: " + ", ".join(dates) + ". "
                "If you want a specific time/day, tell me and I’ll confirm with the instructor.",
                True,
            )
        return (
            "We’re usually free if nothing is scheduled on the calendar, but I must confirm with the instructor first. "
            "Tell me the day/time you want.",
            True,
        )

    # opening hours (set yours via OPENING_HOURS env var)
    if any(k in q for k in ["open", "opening", "hours", "working hours", "what time do you open", "what time are you open"]):
        hours = _env("OPENING_HOURS", default="Please message us to confirm today’s opening hours: https://www.aboodfreediver.com/form1.php")
        return (hours, False)

    # greeting
    if any(k in q for k in ["hello", "hi", "hey", "good morning", "good evening"]):
        return ("Hi, I’m Aqua. Ask me about courses, prices, safety, or availability in Aqaba.", False)

    # default
    return ("Ask me about freediving courses, prices, safety, or availability in Aqaba.", False)


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    session_id = (req.session_id or "").strip() or str(uuid.uuid4())

    # store conversation
    convo = CONVERSATIONS.setdefault(
        session_id,
        {"messages": [], "needs_human": False, "created": datetime.now(timezone.utc).isoformat()},
    )
    convo["messages"].append({"role": "user", "text": question, "ts": datetime.now(timezone.utc).isoformat()})

    # try gemini
    answer = try_gemini_answer(question, req.history)
    if answer:
        needs_human = any(k in question.lower() for k in ["availability", "available", "book", "booking", "reserve", "date", "dates", "schedule"])
        if needs_human:
            convo["needs_human"] = True
            send_owner_email(
                subject="Aqua needs confirmation (availability/booking)",
                body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
            )
        convo["messages"].append({"role": "assistant", "text": answer, "ts": datetime.now(timezone.utc).isoformat()})
        return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source="gemini")

    # fallback
    answer2, needs_human2 = fallback_answer(question)
    if needs_human2:
        convo["needs_human"] = True
        send_owner_email(
            subject="Aqua needs confirmation (availability/booking)",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
        )
    convo["messages"].append({"role": "assistant", "text": answer2, "ts": datetime.now(timezone.utc).isoformat()})
    return ChatResponse(answer=answer2, session_id=session_id, needs_human=needs_human2, source="fallback")


@app.get("/chat/status")
def chat_status(session_id: str):
    convo = CONVERSATIONS.get(session_id)
    if not convo:
        return {"messages": [], "needs_human": False}
    return {"messages": convo["messages"], "needs_human": convo["needs_human"]}


@app.post("/human")
def human_reply(data: HumanReply, _: bool = Depends(admin_auth)):
    convo = CONVERSATIONS.get(data.session_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Session not found")

    convo["messages"].append({"role": "human", "text": data.message, "ts": datetime.now(timezone.utc).isoformat()})
    convo["needs_human"] = False
    return {"ok": True}


@app.get("/admin", response_class=HTMLResponse)
def admin_page(_: bool = Depends(admin_auth)):
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Aqua – Instructor Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: Arial; padding: 20px; max-width: 760px; margin: 0 auto; }
    input, textarea, button { width: 100%; margin: 10px 0; padding: 10px; font-size: 16px; }
    .row { display: grid; grid-template-columns: 1fr; gap: 10px; }
    pre { background: #f5f5f5; padding: 10px; overflow: auto; }
    small { color: #444; }
  </style>
</head>
<body>
  <h2>Reply as Human Instructor</h2>
  <small>Paste the session_id returned by /chat. You can also check messages via /chat/status?session_id=...</small>

  <input id="sid" placeholder="Session ID" />
  <textarea id="msg" rows="4" placeholder="Your reply to the diver"></textarea>
  <button onclick="send()">Send Reply</button>

  <button onclick="loadStatus()">Load chat status</button>

  <pre id="out"></pre>

<script>
async function send() {
  const sid = document.getElementById('sid').value.trim();
  const msg = document.getElementById('msg').value.trim();
  if (!sid || !msg) {
    document.getElementById('out').textContent = "Session ID and message are required.";
    return;
  }
  const res = await fetch('/human', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sid, message: msg })
  });
  document.getElementById('out').textContent = await res.text();
}

async function loadStatus() {
  const sid = document.getElementById('sid').value.trim();
  if (!sid) {
    document.getElementById('out').textContent = "Enter Session ID first.";
    return;
  }
  const res = await fetch('/chat/status?session_id=' + encodeURIComponent(sid));
  document.getElementById('out').textContent = await res.text();
}
</script>
</body>
</html>
"""
