from __future__ import annotations

import os
import re
import uuid
import time
import html as html_lib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin

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
app = FastAPI(title="Aqua – Abood Freediver Assistant", version="1.2.2")

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
    source: str = "gemini"  # "gemini" or "fallback"


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


# -----------------------------
# Admin Auth
# -----------------------------
def admin_auth(credentials: HTTPBasicCredentials = Depends(security)):
    expected_user = _env("ADMIN_USER", "ADMIN_USERNAME")
    expected_pass = _env("ADMIN_PASS", "ADMIN_PASSWORD")

    user_ok = secrets.compare_digest(credentials.username or "", expected_user or "")
    pass_ok = secrets.compare_digest(credentials.password or "", expected_pass or "")

    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


# -----------------------------
# Email notify via SendGrid API (Render-safe: HTTPS)
# -----------------------------
def send_owner_email(subject: str, body: str) -> None:
    """
    Uses SendGrid Email API over HTTPS.
    Required env:
      SENDGRID_API_KEY
      SENDGRID_FROM  (must be a verified Single Sender or authenticated domain sender)
      OWNER_NOTIFY_EMAIL
    """
    owner_to = _env("OWNER_NOTIFY_EMAIL")
    if not owner_to:
        return

    api_key = _env("SENDGRID_API_KEY")
    sg_from = _env("SENDGRID_FROM", "SMTP_FROM")  # allow reusing SMTP_FROM as sender
    if not api_key or not sg_from:
        return

    payload = {
        "personalizations": [{"to": [{"email": owner_to}], "subject": subject}],
        "from": {"email": sg_from},
        "content": [{"type": "text/plain", "value": body}],
    }

    try:
        r = requests.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        if r.status_code != 202:
            print(f"SendGrid FAILED: status={r.status_code} body={r.text[:500]}")
    except Exception as e:
        print(f"SendGrid FAILED: {type(e).__name__}: {e}")


# -----------------------------
# Calendar check (basic scrape + cache)
# -----------------------------
CALENDAR_URL = _env("CALENDAR_URL", default="https://www.aboodfreediver.com/calendar.php")
_calendar_cache: Dict[str, Any] = {"ts": 0.0, "events": []}


def fetch_calendar_events() -> List[str]:
    """
    Tries to extract upcoming event date lines from your public calendar page.
    Cached for 10 minutes.
    """
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
        dates = list(dict.fromkeys(date_regex.findall(html)))[:5]

        _calendar_cache["ts"] = now
        _calendar_cache["events"] = dates
        return dates
    except Exception:
        _calendar_cache["ts"] = now
        _calendar_cache["events"] = []
        return []


# -----------------------------
# Retrieval: site -> blog -> web
# -----------------------------
SITE_BASE_URL = _env("SITE_BASE_URL", default="https://www.aboodfreediver.com")
SITE_SITEMAP_URL = _env("SITE_SITEMAP_URL", default=urljoin(SITE_BASE_URL, "/sitemap.xml"))
BLOG_SITEMAP_URL = _env("BLOG_SITEMAP_URL", default="")

_retrieval_cache: Dict[str, Any] = {
    "ts": 0.0,
    "site_urls": [],
    "blog_urls": [],
}


def _fetch_sitemap_urls(sitemap_url: str, limit: int = 400) -> List[str]:
    if not sitemap_url:
        return []
    try:
        r = requests.get(sitemap_url, timeout=15, headers={"User-Agent": "AquaBot/1.0"})
        r.raise_for_status()
        xml = r.text

        urls = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, flags=re.IGNORECASE)
        urls = [u.strip() for u in urls if u.strip().startswith("http")]

        seen = set()
        out: List[str] = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def _refresh_url_lists_if_needed() -> None:
    now = time.time()
    if now - float(_retrieval_cache.get("ts", 0.0)) < 3600 and _retrieval_cache.get("site_urls"):
        return

    site_urls = _fetch_sitemap_urls(SITE_SITEMAP_URL, limit=500)
    blog_urls: List[str] = _fetch_sitemap_urls(BLOG_SITEMAP_URL, limit=500) if BLOG_SITEMAP_URL else []

    _retrieval_cache["site_urls"] = site_urls
    _retrieval_cache["blog_urls"] = blog_urls
    _retrieval_cache["ts"] = now


def _clean_text_from_html(html_text: str) -> str:
    html_text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
    html_text = re.sub(r"(?is)<style.*?>.*?</style>", " ", html_text)
    text = re.sub(r"(?s)<[^>]+>", " ", html_text)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _score_text(question: str, text: str) -> int:
    q_words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    q_words = [w for w in q_words if len(w) >= 3]
    if not q_words:
        return 0
    t = text.lower()
    return sum(1 for w in q_words if w in t)


def _search_urls_for_context(
    question: str,
    urls: List[str],
    per_url_chars: int = 1400,
    max_hits: int = 4,
    hard_limit_urls: int = 350,
) -> List[Dict[str, str]]:
    hits: List[Tuple[int, str, str]] = []
    for u in urls[:hard_limit_urls]:
        try:
            r = requests.get(u, timeout=12, headers={"User-Agent": "AquaBot/1.0"})
            if r.status_code >= 400:
                continue
            text = _clean_text_from_html(r.text)
            if not text:
                continue
            s = _score_text(question, text)
            if s <= 0:
                continue
            snippet = text[:per_url_chars]
            hits.append((s, u, snippet))
        except Exception:
            continue

    hits.sort(key=lambda x: x[0], reverse=True)
    return [{"url": u, "snippet": snip} for _, u, snip in hits[:max_hits]]


def web_search_context(question: str) -> List[Dict[str, str]]:
    """
    Web search requires ONE option:
      - SERPAPI_KEY
      - GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX
    Returns list of {url, snippet}
    """
    serpapi_key = _env("SERPAPI_KEY")
    if serpapi_key:
        try:
            r = requests.get(
                "https://serpapi.com/search.json",
                timeout=15,
                params={"engine": "google", "q": question, "api_key": serpapi_key, "num": 5},
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("organic_results", [])[:4]
            out: List[Dict[str, str]] = []
            for it in results:
                link = it.get("link")
                snippet = it.get("snippet") or it.get("title") or ""
                if link:
                    out.append({"url": link, "snippet": snippet})
            return out
        except Exception:
            return []

    cse_key = _env("GOOGLE_CSE_API_KEY")
    cse_cx = _env("GOOGLE_CSE_CX")
    if cse_key and cse_cx:
        try:
            r = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                timeout=15,
                params={"key": cse_key, "cx": cse_cx, "q": question, "num": 5},
            )
            r.raise_for_status()
            data = r.json()
            items = data.get("items", [])[:4]
            out: List[Dict[str, str]] = []
            for it in items:
                link = it.get("link")
                snippet = it.get("snippet") or it.get("title") or ""
                if link:
                    out.append({"url": link, "snippet": snippet})
            return out
        except Exception:
            return []

    return []


def retrieve_context(question: str) -> Dict[str, Any]:
    """
    Priority:
      1) SITE (SITE_SITEMAP_URL)
      2) BLOG (BLOG_SITEMAP_URL)
      3) WEB (SerpAPI or Google CSE)
    """
    _refresh_url_lists_if_needed()

    site_hits = _search_urls_for_context(question, _retrieval_cache.get("site_urls", []))
    if site_hits:
        return {"source": "site", "items": site_hits}

    blog_urls = _retrieval_cache.get("blog_urls", [])
    blog_hits = _search_urls_for_context(question, blog_urls) if blog_urls else []
    if blog_hits:
        return {"source": "blog", "items": blog_hits}

    web_hits = web_search_context(question)
    if web_hits:
        return {"source": "web", "items": web_hits}

    return {"source": "none", "items": []}


# -----------------------------
# Gemini (supports either new or old SDK)
# -----------------------------
def try_gemini_answer(question: str, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        return None

    model = _env("GEMINI_MODEL", default="gemini-2.5-flash")

    # calendar context
    dates = fetch_calendar_events()
    cal_context = (
        "Upcoming dates from the calendar: " + ", ".join(dates)
        if dates
        else "Calendar shows no upcoming dates (may mean mostly free; must confirm with instructor)."
    )

    # retrieval context (site -> blog -> web)
    rc = retrieve_context(question)
    context_blocks: List[str] = [cal_context]
    if rc.get("items"):
        label = str(rc.get("source", "context")).upper()
        joined = "\n".join([f"- {it['url']}\n  {it['snippet']}" for it in rc["items"]])
        context_blocks.append(f"{label} CONTEXT:\n{joined}")

    system = (
        "You are Aqua, the freediving assistant for Abood Freediver in Aqaba, Jordan (Red Sea).\n"
        "\n"
        "You MUST follow this priority when answering:\n"
        "1) Use SITE CONTEXT first.\n"
        "2) If insufficient, use BLOG CONTEXT.\n"
        "3) If still insufficient, use WEB CONTEXT.\n"
        "\n"
        "If you use any context, cite the URL(s) you used.\n"
        "If uncertain, ask ONE short follow-up question.\n"
        "Be brief, clear, and helpful.\n"
        "\n"
        "Hard links:\n"
        "- Prices: https://www.aboodfreediver.com/Prices.php?lang=en\n"
        "- Booking/contact form: https://www.aboodfreediver.com/form1.php\n"
    )

    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system + "\n\n" + "\n\n".join(context_blocks)}
    ]

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
        gm = genai_old.GenerativeModel(
            model_name=model,
            system_instruction=system + "\n\n" + "\n\n".join(context_blocks),
        )

        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs if m["role"] != "system"])
        r = gm.generate_content(prompt)
        text = getattr(r, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception:
        return None

    return None


# -----------------------------
# Fallback logic (only if Gemini fails)
# -----------------------------
def fallback_answer(question: str) -> Tuple[str, bool]:
    q = question.strip().lower()
    form_url = "https://www.aboodfreediver.com/form1.php"

    # keep it minimal: Gemini should handle most cases
    if any(k in q for k in ["book", "booking", "reserve", "reservation", "availability", "available", "calendar", "date", "dates"]):
        return (f"Please share the day/time you want. You can also use the booking form: {form_url}", True)

    return ("Please tell me what you need (course, prices, safety, equipment, or availability).", False)


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

    convo = CONVERSATIONS.setdefault(
        session_id,
        {"messages": [], "needs_human": False, "created": datetime.now(timezone.utc).isoformat()},
    )
    convo["messages"].append({"role": "user", "text": question, "ts": datetime.now(timezone.utc).isoformat()})

    q = question.lower().strip()

    # Gemini answers EVERYTHING (site -> blog -> web context is injected inside try_gemini_answer)
    answer = try_gemini_answer(question, req.history)
    if answer:
        needs_human = any(
            k in q
            for k in [
                "availability", "available", "calendar", "date", "dates", "schedule", "time slot",
                "book", "booking", "reserve", "reservation",
            ]
        )

        if needs_human:
            convo["needs_human"] = True
            send_owner_email(
                subject="Aqua needs confirmation (availability/booking)",
                body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
            )

        convo["messages"].append({"role": "assistant", "text": answer, "ts": datetime.now(timezone.utc).isoformat()})
        return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source="gemini")

    # fallback only if Gemini fails completely
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
