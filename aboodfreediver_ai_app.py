from __future__ import annotations

import os
import re
import uuid
import time
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
app = FastAPI(title="Aqua – Abood Freediver Assistant", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: lock down in production
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


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------
# Admin Auth
# -----------------------------
def admin_auth(credentials: HTTPBasicCredentials = Depends(security)):
    expected_user = _env("ADMIN_USER", "ADMIN_USERNAME")
    expected_pass = _env("ADMIN_PASS", "ADMIN_PASSWORD")

    # If env vars are not set, block access (avoid accidental open admin)
    if not expected_user or not expected_pass:
        raise HTTPException(status_code=500, detail="Admin credentials not configured")

    user_ok = secrets.compare_digest(credentials.username or "", expected_user)
    pass_ok = secrets.compare_digest(credentials.password or "", expected_pass)

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


# -----------------------------
# Email notify via SendGrid API (NO SMTP)
# -----------------------------
SENDGRID_API_URL = "https://api.sendgrid.com/v3/mail/send"


def send_owner_email(subject: str, body: str) -> None:
    """
    Uses SendGrid Email API (works on Render).
    Env vars:
      SENDGRID_API_KEY (required)
      SMTP_FROM or SEND_FROM_EMAIL (sender email)
      SMTP_FROM_NAME or SEND_FROM_NAME (optional)
      OWNER_NOTIFY_EMAIL (required)
    """
    owner_to = _env("OWNER_NOTIFY_EMAIL")
    api_key = _env("SENDGRID_API_KEY")

    # "SMTP_FROM" already exists in your Render env screenshot; reuse it
    from_email = _env("SMTP_FROM", "SEND_FROM_EMAIL")
    from_name = _env("SMTP_FROM_NAME", "SEND_FROM_NAME", default="Aqua Assistant")

    if not owner_to or not api_key or not from_email:
        return

    payload = {
        "personalizations": [{"to": [{"email": owner_to}]}],
        "from": {"email": from_email, "name": from_name},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }

    try:
        r = requests.post(
            SENDGRID_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        # 202 Accepted is success for SendGrid
        if r.status_code not in (200, 202):
            print(f"SendGrid email FAILED: {r.status_code} {r.text}")
    except Exception as e:
        print(f"SendGrid email FAILED: {type(e).__name__}: {e}")


# -----------------------------
# Calendar check (basic scrape + cache)
# -----------------------------
CALENDAR_URL = _env("CALENDAR_URL", default="https://www.aboodfreediver.com/calender.php")
_calendar_cache: Dict[str, Any] = {"ts": 0.0, "events": []}


def fetch_calendar_events() -> List[str]:
    """
    Extract upcoming event date lines from public calendar page.
    Cached for 10 minutes to avoid hammering site.
    """
    now = time.time()
    if now - float(_calendar_cache.get("ts", 0.0)) < 600 and isinstance(_calendar_cache.get("events"), list):
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

    # --- Try new SDK first: google-genai ---
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

    # --- Fallback old SDK: google-generativeai ---
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

    if any(k in q for k in ["price", "prices", "cost", "how much", "fee"]):
        return ("Prices are here: https://www.aboodfreediver.com/Prices.php?lang=en", False)

    if any(k in q for k in ["book", "booking", "reserve", "reservation"]):
        return (
            f"To book, please fill this form: {FORM_URL} "
            "(or tell me the date/time you want and I will confirm availability).",
            True,
        )

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

    if any(
        k in q
        for k in [
            "availability",
            "available",
            "calendar",
            "date",
            "dates",
            "schedule",
            "time slot",
            "time are you free",
        ]
    ):
        dates = fetch_calendar_events()
        if dates:
            return (
                "Upcoming calendar dates: " + ", ".join(dates) + ". "
                "Tell me the day/time you want and I’ll confirm with the instructor.",
                True,
            )
        return (
            "We’re usually free if nothing is scheduled on the calendar, but I must confirm with the instructor first. "
            "Tell me the day/time you want.",
            True,
        )

    if any(k in q for k in ["open", "opening", "hours", "working hours", "what time do you open", "what time are you open"]):
        hours = _env(
            "OPENING_HOURS",
            default=f"Opening hours: please use the contact form to confirm today: {FORM_URL}",
        )
        return (hours, False)

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

    if any(k in q for k in ["hello", "hi", "hey", "good morning", "good evening"]):
        return ("Hi, I’m Aqua. Ask me about courses, prices, safety, or availability in Aqaba.", False)

    return ("Ask me about freediving courses, prices, safety, or availability in Aqaba.", False)


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": _utcnow_iso()}


@app.get("/")
def root():
    # Render health checks often hit "/"
    return {"ok": True, "service": "Aqua backend", "time": _utcnow_iso()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    session_id = (req.session_id or "").strip() or str(uuid.uuid4())

    convo = CONVERSATIONS.setdefault(
        session_id,
        {"messages": [], "needs_human": False, "created": _utcnow_iso()},
    )
    convo["messages"].append({"role": "user", "text": question, "ts": _utcnow_iso()})

    q = question.lower().strip()
    FORM_URL = "https://www.aboodfreediver.com/form1.php"

    # --- RULE OVERRIDES (NO NOTIFY) ---
    if any(k in q for k in ["open", "opening", "hours", "working hours", "what time do you open", "what time are you open"]):
        hours = _env("OPENING_HOURS", default="Open daily 09:00–17:00 (Aqaba time).")
        convo["messages"].append({"role": "assistant", "text": hours, "ts": _utcnow_iso()})
        return ChatResponse(answer=hours, session_id=session_id, needs_human=False, source="rules")

    if any(k in q for k in ["whatsapp", "phone", "contact", "email", "number"]):
        phone = _env("CONTACT_PHONE", default="")
        email = _env("CONTACT_EMAIL", default="free@aboodfreediver.com")
        whatsapp = _env("CONTACT_WHATSAPP", default=phone)

        msg = "You can contact us here:\n"
        if whatsapp:
            msg += f"Phone/WhatsApp: {whatsapp}\n"
        msg += f"Email: {email}\n"
        msg += f"Form: {FORM_URL}"

        convo["messages"].append({"role": "assistant", "text": msg, "ts": _utcnow_iso()})
        return ChatResponse(answer=msg, session_id=session_id, needs_human=False, source="rules")

    if any(k in q for k in ["courses", "course", "levels", "learn", "training", "certification"]):
        msg = (
            "We offer freediving courses for all levels:\n"
            "- Discovery Freediver (beginner try)\n"
            "- Freediver (Level 1)\n"
            "- Advanced Freediver (Level 2)\n"
            "- Master Freediver (Level 3)\n\n"
            "Tell me your experience level and how many days you have, and I’ll recommend the best option."
        )
        convo["messages"].append({"role": "assistant", "text": msg, "ts": _utcnow_iso()})
        return ChatResponse(answer=msg, session_id=session_id, needs_human=False, source="rules")
    # --- END RULE OVERRIDES ---

    answer = try_gemini_answer(question, req.history)
    if answer:
        needs_human = any(
            k in q
            for k in [
                "availability",
                "available",
                "calendar",
                "date",
                "dates",
                "schedule",
                "time slot",
                "book",
                "booking",
                "reserve",
                "reservation",
            ]
        )

        if needs_human:
            convo["needs_human"] = True
            send_owner_email(
                subject="Aqua needs confirmation (availability/booking)",
                body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
            )

        convo["messages"].append({"role": "assistant", "text": answer, "ts": _utcnow_iso()})
        return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source="gemini")

    answer2, needs_human2 = fallback_answer(question)
    if needs_human2:
        convo["needs_human"] = True
        send_owner_email(
            subject="Aqua needs confirmation (availability/booking)",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
        )

    convo["messages"].append({"role": "assistant", "text": answer2, "ts": _utcnow_iso()})
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

    convo["messages"].append({"role": "human", "text": data.message, "ts": _utcnow_iso()})
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
