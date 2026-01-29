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
app = FastAPI(title="Aqua – Abood Freediver Assistant", version="1.2.1")

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
    source: str = "fallback"  # "gemini" or "fallback" or "rules"


class HumanReply(BaseModel):
    session_id: str
    message: str


# -----------------------------
# Helpers
# -----------------------------
def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v)
    return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_message(session_id: str, role: str, text: str) -> None:
    convo = CONVERSATIONS.setdefault(
        session_id,
        {"messages": [], "needs_human": False, "created": _now_iso()},
    )
    convo["messages"].append({"role": role, "text": text, "ts": _now_iso()})


def _finalize_response(
    session_id: str,
    answer: str,
    needs_human: bool,
    source: str,
) -> ChatResponse:
    # store assistant message BEFORE returning
    _add_message(session_id, "assistant", answer)

    if needs_human:
        convo = CONVERSATIONS.setdefault(
            session_id,
            {"messages": [], "needs_human": False, "created": _now_iso()},
        )
        convo["needs_human"] = True

    return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source=source)


def _is_booking_or_availability(q: str) -> bool:
    q = q.lower()

    booking_keywords = [
        "book", "booking", "reserve", "reservation", "register", "sign up", "join",
        "price for tomorrow",  # common phrasing
    ]
    availability_keywords = [
        "availability", "available", "calendar", "schedule", "time slot",
        "date", "dates", "when", "are you free",
        "today", "tomorrow", "tonight", "this week", "next week", "weekend",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    ]

    # If user mentions booking words OR time/date intent => needs human confirmation
    if any(k in q for k in booking_keywords):
        return True
    if any(k in q for k in availability_keywords) and any(
        x in q for x in ["book", "course", "session", "training", "class", "fun", "discover", "freedive", "freediving"]
    ):
        return True

    # Many users write: "I want to book a freediving course tomorrow"
    # This contains "course" and "tomorrow" but might miss booking word if typo.
    if "tomorrow" in q and any(x in q for x in ["course", "session", "training", "freedive", "freediving"]):
        return True

    return False


# -----------------------------
# Admin Auth
# -----------------------------
def admin_auth(credentials: HTTPBasicCredentials = Depends(security)):
    expected_user = _env("ADMIN_USER", "ADMIN_USERNAME")
    expected_pass = _env("ADMIN_PASS", "ADMIN_PASSWORD")

    user_ok = secrets.compare_digest(credentials.username or "", expected_user or "")
    pass_ok = secrets.compare_digest(credentials.password or "", expected_pass or "")

    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True


# -----------------------------
# Email notify
# -----------------------------
def send_owner_email(subject: str, body: str) -> None:
    """
    Uses SMTP_* env vars:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM
    Sends to OWNER_NOTIFY_EMAIL.

    For Gmail (recommended):
      SMTP_HOST=smtp.gmail.com
      SMTP_PORT=587
      SMTP_USER=aboodfreediver@gmail.com
      SMTP_PASSWORD=<GMAIL_APP_PASSWORD>   (NOT your normal password)
      SMTP_FROM=aboodfreediver@gmail.com
      OWNER_NOTIFY_EMAIL=aboodfreediver@gmail.com
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
    except Exception as e:
        print(f"Email notification FAILED: {type(e).__name__}: {e}")
        return


# -----------------------------
# Calendar check (scrape + cache)
# -----------------------------
CALENDAR_URL = _env("CALENDAR_URL", default="https://www.aboodfreediver.com/calender.php")
_calendar_cache: Dict[str, Any] = {"ts": 0.0, "events": []}


def fetch_calendar_events() -> List[str]:
    now = time.time()
    if now - _calendar_cache["ts"] < 600 and isinstance(_calendar_cache.get("events"), list):
        return _calendar_cache["events"]

    try:
        r = requests.get(CALENDAR_URL, timeout=10)
        r.raise_for_status()
        html = r.text

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
# Gemini
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

    dates = fetch_calendar_events()
    if dates:
        cal_context = "Upcoming dates from the calendar: " + ", ".join(dates)
    else:
        cal_context = "Calendar shows no upcoming dates (may mean mostly free; must confirm with instructor)."

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system + "\n\n" + cal_context}]

    if history:
        for m in history[-12:]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": question})

    # New SDK: google-genai
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs])
        resp = client.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception:
        pass

    # Old SDK: google-generativeai
    try:
        import google.generativeai as genai_old  # type: ignore

        genai_old.configure(api_key=api_key)
        gm = genai_old.GenerativeModel(model_name=model, system_instruction=system + "\n\n" + cal_context)

        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs if m["role"] != "system"])
        r = gm.generate_content(prompt)
        text = getattr(r, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception:
        return None

    return None


# -----------------------------
# Fallback logic
# -----------------------------
def fallback_answer(question: str) -> Tuple[str, bool]:
    q = question.strip().lower()
    FORM_URL = "https://www.aboodfreediver.com/form1.php"

    # booking / availability must be checked BEFORE "courses"
    if _is_booking_or_availability(q):
        dates = fetch_calendar_events()
        if dates:
            return (
                "To book, please fill this form: "
                f"{FORM_URL}\n\n"
                "Upcoming calendar dates: " + ", ".join(dates) + ".\n"
                "Tell me the day/time you want and I’ll confirm with the instructor.",
                True,
            )
        return (
            "To book, please fill this form: "
            f"{FORM_URL}\n\n"
            "We’re usually free if nothing is scheduled on the calendar, but I must confirm with the instructor first.\n"
            "Tell me the day/time you want.",
            True,
        )

    # prices
    if any(k in q for k in ["price", "prices", "cost", "how much", "fee"]):
        return ("Prices are here: https://www.aboodfreediver.com/Prices.php?lang=en", False)

    # contact info
    if any(k in q for k in ["whatsapp", "phone", "contact", "email", "number"]):
        phone = _env("CONTACT_PHONE", default="")
        email = _env("CONTACT_EMAIL", default="free@aboodfreediver.com")
        whatsapp = _env("CONTACT_WHATSAPP", default=phone)

        msg = "You can contact us here:\n"
        if whatsapp:
            msg += f"Phone/WhatsApp: {whatsapp}\n"
        msg += f"Email: {email}\n"
        msg += f"Form: {FORM_URL}"
        return (msg, False)

    # opening hours
    if any(k in q for k in ["open", "opening", "hours", "working hours", "what time do you open", "what time are you open"]):
        hours = _env(
            "OPENING_HOURS",
            default=f"Opening hours: please use the contact form to confirm today: {FORM_URL}",
        )
        return (hours, False)

    # courses (clear answer)
    if any(k in q for k in ["courses", "course", "levels", "learn", "training", "certification"]):
        return (
            "We offer freediving courses for all levels:\n"
            "- Discovery Freediver (beginner try)\n"
            "- Freediver (Level 1)\n"
            "- Advanced Freediver (Level 2)\n"
            "- Master Freediver (Level 3)\n\n"
            "Tell me your experience level and how many days you have, and I’ll recommend the best option.",
            False,
        )

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
    return {"ok": True, "time": _now_iso()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    session_id = (req.session_id or "").strip() or str(uuid.uuid4())

    # store user message
    _add_message(session_id, "user", question)

    q = question.lower().strip()
    FORM_URL = "https://www.aboodfreediver.com/form1.php"

    # 1) HARD RULE: booking/availability ALWAYS needs human + email
    # This prevents Gemini from answering a "courses list" when user wants to book tomorrow.
    if _is_booking_or_availability(q):
        dates = fetch_calendar_events()
        if dates:
            answer = (
                f"To book, please fill this form: {FORM_URL}\n\n"
                "Upcoming calendar dates: " + ", ".join(dates) + ".\n"
                "Tell me the day/time you want and I’ll confirm with the instructor."
            )
        else:
            answer = (
                f"To book, please fill this form: {FORM_URL}\n\n"
                "We’re usually free if nothing is scheduled on the calendar, but I must confirm with the instructor first.\n"
                "Tell me the day/time you want."
            )

        # mark + email notify
        CONVERSATIONS.setdefault(session_id, {"messages": [], "needs_human": False, "created": _now_iso()})["needs_human"] = True
        send_owner_email(
            subject="Aqua needs confirmation (availability/booking)",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
        )
        return _finalize_response(session_id, answer, needs_human=True, source="rules")

    # 2) RULE OVERRIDES (NO NOTIFY)
    if any(k in q for k in ["open", "opening", "hours", "working hours", "what time do you open", "what time are you open"]):
        hours = _env("OPENING_HOURS", default="Open daily 9:00-17:00 (Aqaba time).")
        return _finalize_response(session_id, hours, needs_human=False, source="rules")

    if any(k in q for k in ["whatsapp", "phone", "contact", "email", "number"]):
        phone = _env("CONTACT_PHONE", default="")
        email = _env("CONTACT_EMAIL", default="free@aboodfreediver.com")
        whatsapp = _env("CONTACT_WHATSAPP", default=phone)

        msg = "You can contact us here:\n"
        if whatsapp:
            msg += f"Phone/WhatsApp: {whatsapp}\n"
        msg += f"Email: {email}\n"
        msg += f"Form: {FORM_URL}"
        return _finalize_response(session_id, msg, needs_human=False, source="rules")

    if any(k in q for k in ["courses", "course", "levels", "learn", "training", "certification"]):
        msg = (
            "We offer freediving courses for all levels:\n"
            "- Discovery Freediver (beginner try)\n"
            "- Freediver (Level 1)\n"
            "- Advanced Freediver (Level 2)\n"
            "- Master Freediver (Level 3)\n\n"
            "Tell me your experience level and how many days you have, and I’ll recommend the best option."
        )
        return _finalize_response(session_id, msg, needs_human=False, source="rules")

    # 3) Try Gemini
    answer = try_gemini_answer(question, req.history)
    if answer:
        # Gemini normal answers do not trigger notify unless booking/availability (already handled above)
        return _finalize_response(session_id, answer, needs_human=False, source="gemini")

    # 4) fallback
    answer2, needs_human2 = fallback_answer(question)
    if needs_human2:
        CONVERSATIONS.setdefault(session_id, {"messages": [], "needs_human": False, "created": _now_iso()})["needs_human"] = True
        send_owner_email(
            subject="Aqua needs confirmation (availability/booking)",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
        )
    return _finalize_response(session_id, answer2, needs_human=needs_human2, source="fallback")


@app.get("/chat/status")
def chat_status(session_id: str):
    convo = CONVERSATIONS.get(session_id)
    if not convo:
        return {"messages": [], "needs_human": False}
    return {"messages": convo["messages"], "needs_human": bool(convo.get("needs_human"))}


@app.post("/human")
def human_reply(data: HumanReply, _: bool = Depends(admin_auth)):
    convo = CONVERSATIONS.get(data.session_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Session not found")

    _add_message(data.session_id, "human", data.message)
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
