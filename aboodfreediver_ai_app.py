from __future__ import annotations

import os
import re
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import secrets
import requests
from urllib.parse import urljoin
import html as html_lib

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

# -----------------------------
# Models
# -----------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    needs_human: bool = False
    source: str = "site"  # "site" | "web" | "gemini" | "fallback" | "rules"


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


def _strip_html_to_text(raw_html: str) -> str:
    raw_html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw_html)
    raw_html = re.sub(r"(?is)<[^>]+>", " ", raw_html)
    raw_html = html_lib.unescape(raw_html)
    raw_html = re.sub(r"\s+", " ", raw_html).strip()
    return raw_html


# -----------------------------
# Site config
# -----------------------------
BASE_SITE = _env("BASE_SITE_URL", "SITE_BASE_URL", default="https://www.aboodfreediver.com/")
if not BASE_SITE.endswith("/"):
    BASE_SITE += "/"

SITE_SITEMAP_URL = _env("SITE_SITEMAP_URL", default=urljoin(BASE_SITE, "sitemaps.XML"))
BLOG_INDEX_URL = _env("BLOG_SITEMAP_URL", default=urljoin(BASE_SITE, "blog.html"))

_PAGE_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 60 * 60  # 1 hour

_SITEMAP_CACHE: Dict[str, Any] = {"ts": 0.0, "urls": []}
_SITEMAP_TTL_SECONDS = 6 * 60 * 60  # 6 hours


def fetch_page_text(url: str, timeout: int = 12) -> str:
    now = time.time()
    cached = _PAGE_CACHE.get(url)
    if cached and (now - float(cached.get("ts", 0)) < _CACHE_TTL_SECONDS):
        return str(cached.get("text", ""))

    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "AquaBot/1.0"})
        r.raise_for_status()
        text = _strip_html_to_text(r.text)
    except Exception:
        text = ""

    _PAGE_CACHE[url] = {"ts": now, "text": text}
    return text


def fetch_sitemap_urls(limit: int = 250) -> List[str]:
    """
    Uses your SITE_SITEMAP_URL to discover pages.
    Caches results to avoid hammering your site.
    """
    now = time.time()
    if now - float(_SITEMAP_CACHE.get("ts", 0)) < _SITEMAP_TTL_SECONDS and isinstance(_SITEMAP_CACHE.get("urls"), list):
        return list(_SITEMAP_CACHE["urls"])

    urls: List[str] = []
    try:
        r = requests.get(SITE_SITEMAP_URL, timeout=15, headers={"User-Agent": "AquaBot/1.0"})
        r.raise_for_status()
        xml = r.text
        # typical sitemap: <loc>https://...</loc>
        locs = re.findall(r"(?is)<loc>\s*([^<\s]+)\s*</loc>", xml)
        for u in locs:
            u = u.strip()
            if u.startswith(BASE_SITE):
                urls.append(u)
    except Exception:
        urls = []

    # de-dup, cap
    urls = list(dict.fromkeys(urls))[:limit]
    _SITEMAP_CACHE["ts"] = now
    _SITEMAP_CACHE["urls"] = urls
    return urls


def extract_links(html_text: str, base_url: str) -> List[str]:
    links = re.findall(r'(?i)href\s*=\s*["\']([^"\']+)["\']', html_text)
    out: List[str] = []
    for href in links:
        href = href.strip()
        if not href or href.startswith("#"):
            continue
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        full = urljoin(base_url, href)
        if full.startswith(BASE_SITE):
            out.append(full)
    return list(dict.fromkeys(out))


def get_blog_urls(max_posts: int = 50) -> List[str]:
    try:
        r = requests.get(BLOG_INDEX_URL, timeout=12, headers={"User-Agent": "AquaBot/1.0"})
        r.raise_for_status()
        links = extract_links(r.text, BLOG_INDEX_URL)
        blog_links = [u for u in links if re.search(r"/blog\d+\.html$", u, re.I)]
        return blog_links[:max_posts]
    except Exception:
        return []


def score_text_match(query: str, text: str) -> int:
    q_terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 3]
    if not q_terms:
        return 0
    t = text.lower()
    score = 0
    for term in set(q_terms):
        if term in t:
            score += 2
    # bonus if page contains "pre-requisites"/"prerequisites"
    if "pre-requisites" in t or "prerequisites" in t:
        score += 1
    return score


def extract_best_sentences(question: str, text: str, max_sentences: int = 2) -> str:
    """
    Pulls a short direct answer from page text: choose sentences containing query terms.
    """
    q_terms = [t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) >= 3]
    if not q_terms:
        return ""

    # sentence split (simple)
    sentences = re.split(r"(?<=[\.\?\!])\s+", text)
    scored: List[Tuple[int, str]] = []
    for s in sentences:
        sl = s.lower()
        hits = sum(1 for t in set(q_terms) if t in sl)
        if hits > 0 and 25 <= len(s) <= 220:
            scored.append((hits, s.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)

    picked = []
    for _, s in scored[:max_sentences]:
        picked.append(s)
    return " ".join(picked).strip()


def site_answer(question: str, max_pages_to_scan: int = 18) -> Optional[str]:
    """
    Searches your site pages and extracts a direct answer.
    """
    urls = fetch_sitemap_urls(limit=250)
    # add blog pages too (optional)
    urls += get_blog_urls(max_posts=30)
    urls = list(dict.fromkeys([u for u in urls if u.startswith(BASE_SITE)]))

    # Score URLs quickly by scanning a small chunk of each page text
    scored_urls: List[Tuple[int, str]] = []
    for u in urls[:250]:
        txt = fetch_page_text(u)
        if not txt:
            continue
        s = score_text_match(question, txt)
        if s > 0:
            scored_urls.append((s, u))
    scored_urls.sort(key=lambda x: x[0], reverse=True)

    # Scan top N pages and try extracting an answer
    for _, u in scored_urls[:max_pages_to_scan]:
        txt = fetch_page_text(u)
        if not txt:
            continue

        # special strong extraction for age/prerequisites
        ql = question.lower()
        if any(k in ql for k in ["age", "years old", "minimum age", "prerequisite", "pre-requisite", "required"]):
            # try to find an explicit "Be XX years" / "Be at least XX years"
            m = re.search(r"(?i)\bBe\s+(?:at\s+least\s+)?(\d{1,2})\s+years?\b", txt)
            if m:
                age = m.group(1)
                # build a clean direct answer
                return f"Minimum age requirement is {age} years old."

        snippet = extract_best_sentences(question, txt, max_sentences=2)
        if snippet:
            # Make it cleaner (remove repeated nav/footer-like fragments)
            snippet = re.sub(r"\s*(Contact us|Calendar|Prices)\s*", " ", snippet, flags=re.I).strip()
            return snippet

    return None


# -----------------------------
# Web search (optional)
# -----------------------------
def searchapi_web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    api_key = _env("SEARCHAPI_KEY")
    if not api_key:
        return []
    try:
        url = "https://www.searchapi.io/api/v1/search"
        params = {"engine": "google", "q": query, "api_key": api_key, "hl": "en", "gl": "us", "page": 1}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        organic = data.get("organic_results") or []
        out: List[Dict[str, str]] = []
        for item in organic[:k]:
            out.append(
                {"title": str(item.get("title", "")), "link": str(item.get("link", "")), "snippet": str(item.get("snippet", ""))}
            )
        return out
    except Exception:
        return []


def web_answer(question: str) -> Optional[str]:
    """
    Very light web fallback: if SearchAPI returns a good snippet, answer with that snippet.
    """
    results = searchapi_web_search(question, k=5)
    for r in results:
        snip = (r.get("snippet") or "").strip()
        if len(snip) >= 30:
            return snip
    return None


# -----------------------------
# Admin Auth
# -----------------------------
def admin_auth(credentials: HTTPBasicCredentials = Depends(security)):
    expected_user = _env("ADMIN_USER", "ADMIN_USERNAME")
    expected_pass = _env("ADMIN_PASS", "ADMIN_PASSWORD")

    user_ok = secrets.compare_digest(credentials.username or "", expected_user or "")
    pass_ok = secrets.compare_digest(credentials.password or "", expected_pass or "")

    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})
    return True


# -----------------------------
# Email notify via SendGrid API
# -----------------------------
def send_owner_email(subject: str, body: str) -> None:
    owner_to = _env("OWNER_NOTIFY_EMAIL")
    if not owner_to:
        return

    api_key = _env("SENDGRID_API_KEY")
    sg_from = _env("SENDGRID_FROM", "SMTP_FROM")
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
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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
CALENDAR_URL = _env("CALENDAR_URL", default=urljoin(BASE_SITE, "calendar.php"))
_calendar_cache: Dict[str, Any] = {"ts": 0.0, "events": []}


def fetch_calendar_events() -> List[str]:
    now = time.time()
    if now - float(_calendar_cache.get("ts", 0)) < 600 and isinstance(_calendar_cache.get("events"), list):
        return list(_calendar_cache["events"])

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
# Minimal fallback
# -----------------------------
def fallback_answer(question: str) -> str:
    q = question.strip().lower()
    if any(k in q for k in ["price", "prices", "cost", "how much", "fee"]):
        return "Prices are here: https://www.aboodfreediver.com/Prices.php?lang=en"
    if any(k in q for k in ["whatsapp", "phone", "contact", "email", "number"]):
        phone = _env("CONTACT_PHONE", default="")
        email = _env("CONTACT_EMAIL", default="free@aboodfreediver.com")
        whatsapp = _env("CONTACT_WHATSAPP", default=phone)
        msg = "You can contact us here:\n"
        if whatsapp:
            msg += f"Phone/WhatsApp: {whatsapp}\n"
        msg += f"Email: {email}\n"
        msg += "Form: https://www.aboodfreediver.com/form1.php"
        return msg
    return "Please tell me what you want to do (beginner try, course, fun dive, or training days) and I’ll recommend the best option."


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

    # Human-needed detection (booking/availability)
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
            "confirm",
        ]
    )

    # 1) Try answer from your website (DIRECT answer, no links)
    ans = site_answer(question)
    if ans:
        if needs_human:
            convo["needs_human"] = True
            send_owner_email(
                subject="Aqua needs confirmation (booking/availability)",
                body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nOpen admin:\n/admin",
            )
        convo["messages"].append({"role": "assistant", "text": ans, "ts": datetime.now(timezone.utc).isoformat()})
        return ChatResponse(answer=ans, session_id=session_id, needs_human=needs_human, source="site")

    # 2) Calendar quick answer (still direct)
    if any(k in q for k in ["availability", "available", "calendar", "date", "dates", "schedule"]):
        dates = fetch_calendar_events()
        if dates:
            ans2 = "Upcoming dates: " + ", ".join(dates) + ". Tell me your preferred day/time and I’ll confirm."
        else:
            ans2 = "We’re usually free if nothing is scheduled, but I must confirm with the instructor. Tell me your preferred day/time."
        convo["needs_human"] = True
        send_owner_email(
            subject="Aqua needs confirmation (availability)",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nOpen admin:\n/admin",
        )
        convo["messages"].append({"role": "assistant", "text": ans2, "ts": datetime.now(timezone.utc).isoformat()})
        return ChatResponse(answer=ans2, session_id=session_id, needs_human=True, source="rules")

    # 3) Web fallback (direct snippet)
    ans3 = web_answer(question)
    if ans3:
        if needs_human:
            convo["needs_human"] = True
            send_owner_email(
                subject="Aqua needs confirmation (booking/availability)",
                body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nOpen admin:\n/admin",
            )
        convo["messages"].append({"role": "assistant", "text": ans3, "ts": datetime.now(timezone.utc).isoformat()})
        return ChatResponse(answer=ans3, session_id=session_id, needs_human=needs_human, source="web")

    # 4) Final fallback
    ans4 = fallback_answer(question)
    if needs_human:
        convo["needs_human"] = True
        send_owner_email(
            subject="Aqua needs confirmation (booking/availability)",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nOpen admin:\n/admin",
        )
    convo["messages"].append({"role": "assistant", "text": ans4, "ts": datetime.now(timezone.utc).isoformat()})
    return ChatResponse(answer=ans4, session_id=session_id, needs_human=needs_human, source="fallback")


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
