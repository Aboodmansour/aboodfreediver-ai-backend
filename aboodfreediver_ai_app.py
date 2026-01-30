from __future__ import annotations

import os
import re
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin

import secrets
import requests
import html as html_lib
import xml.etree.ElementTree as ET

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field


# =============================================================================
# App
# =============================================================================
app = FastAPI(title="Aqua – Abood Freediver Assistant", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

# In-memory (Render free tier resets on sleep/deploy)
CONVERSATIONS: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# Models
# =============================================================================
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    needs_human: bool = False
    source: str = "fallback"  # gemini | fallback | rules


class HumanReply(BaseModel):
    session_id: str
    message: str


# =============================================================================
# Helpers
# =============================================================================
def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


# =============================================================================
# Config
# =============================================================================
BASE_SITE = _env("SITE_BASE_URL", "BASE_SITE_URL", default="https://www.aboodfreediver.com/")
if not BASE_SITE.endswith("/"):
    BASE_SITE += "/"

SITE_SITEMAP_URL = _env("SITE_SITEMAP_URL", default=urljoin(BASE_SITE, "sitemaps.xml"))
BLOG_INDEX_URL = _env("BLOG_SITEMAP_URL", default=urljoin(BASE_SITE, "blog.html"))

# Networking (keep fast to avoid Render timeouts)
HTTP_TIMEOUT = int(_env("HTTP_TIMEOUT", default="10"))
MAX_FETCH_PAGES_PER_QUESTION = int(_env("MAX_FETCH_PAGES_PER_QUESTION", default="8"))

# Caches
_PAGE_CACHE: Dict[str, Dict[str, Any]] = {}     # url -> {"ts": float, "text": str}
_URLS_CACHE: Dict[str, Any] = {"ts": 0.0, "urls": []}  # {"ts": float, "urls": List[str]}
_CACHE_TTL_PAGE_SECONDS = 60 * 60               # 1 hour
_CACHE_TTL_URLS_SECONDS = 12 * 60 * 60          # 12 hours


# =============================================================================
# SearchApi (Web search fallback)
# =============================================================================
def searchapi_web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    api_key = _env("SEARCHAPI_KEY")
    if not api_key:
        return []

    try:
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "hl": "en",
            "gl": "us",
            "page": 1,
        }
        r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json() or {}

        organic = data.get("organic_results") or []
        out: List[Dict[str, str]] = []
        for item in organic[:k]:
            out.append(
                {
                    "title": str(item.get("title", "")),
                    "link": str(item.get("link", "")),
                    "snippet": str(item.get("snippet", "")),
                }
            )
        return out
    except Exception:
        return []


# =============================================================================
# HTML/Text helpers
# =============================================================================
def _strip_html_to_text(raw_html: str) -> str:
    raw_html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw_html)
    raw_html = re.sub(r"(?is)<[^>]+>", " ", raw_html)
    raw_html = html_lib.unescape(raw_html)
    raw_html = re.sub(r"\s+", " ", raw_html).strip()
    return raw_html


def fetch_page_text(url: str) -> str:
    now = time.time()
    cached = _PAGE_CACHE.get(url)
    if cached and (now - float(cached.get("ts", 0)) < _CACHE_TTL_PAGE_SECONDS):
        return str(cached.get("text", ""))

    text = ""
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": "AquaBot/1.3"})
        r.raise_for_status()
        text = _strip_html_to_text(r.text)
    except Exception:
        text = ""

    _PAGE_CACHE[url] = {"ts": now, "text": text}
    return text


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
        # keep only site pages
        if full.startswith(BASE_SITE):
            out.append(full.split("#")[0])
    # unique preserve order
    return list(dict.fromkeys(out))


# =============================================================================
# URL discovery (Sitemap first, then fallback crawl)
# =============================================================================
def _parse_sitemap_urls(xml_text: str) -> List[str]:
    urls: List[str] = []
    try:
        root = ET.fromstring(xml_text)
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"
        for loc in root.iter(f"{ns}loc"):
            if loc.text:
                u = loc.text.strip()
                if u.startswith(BASE_SITE):
                    urls.append(u.split("#")[0])
    except Exception:
        return []
    return urls


def get_site_urls() -> List[str]:
    now = time.time()
    if _URLS_CACHE["urls"] and (now - float(_URLS_CACHE["ts"]) < _CACHE_TTL_URLS_SECONDS):
        return list(_URLS_CACHE["urls"])

    urls: List[str] = []

    # 1) Sitemap
    try:
        r = requests.get(SITE_SITEMAP_URL, timeout=HTTP_TIMEOUT, headers={"User-Agent": "AquaBot/1.3"})
        r.raise_for_status()
        urls = _parse_sitemap_urls(r.text)
    except Exception:
        urls = []

    # 2) If sitemap empty, crawl homepage + blog index
    if not urls:
        seed_pages = [
            urljoin(BASE_SITE, "index.html"),
            BLOG_INDEX_URL,
            urljoin(BASE_SITE, "Prices.php?lang=en"),
            urljoin(BASE_SITE, "form1.php"),
        ]
        discovered: List[str] = []
        for sp in seed_pages:
            try:
                r = requests.get(sp, timeout=HTTP_TIMEOUT, headers={"User-Agent": "AquaBot/1.3"})
                r.raise_for_status()
                discovered.extend(extract_links(r.text, sp))
            except Exception:
                continue
        urls = list(dict.fromkeys([u for u in discovered if u.startswith(BASE_SITE)]))

    # keep only html/php pages (avoid images, etc.)
    urls = [
        u for u in urls
        if re.search(r"\.(html?|php)(\?|$)", u, re.I)
    ]
    urls = list(dict.fromkeys(urls))

    _URLS_CACHE["ts"] = now
    _URLS_CACHE["urls"] = urls
    return urls


# =============================================================================
# Retrieval (fast)
# =============================================================================
def _query_terms(q: str) -> List[str]:
    terms = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) >= 3]
    # de-dup keep order
    seen = set()
    out = []
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _url_relevance_score(terms: List[str], url: str) -> int:
    u = url.lower()
    score = 0
    for t in terms:
        if t in u:
            score += 3
    # Boost obvious course pages for course-like questions
    if any(t in terms for t in ["course", "courses", "freediver", "discover", "advanced", "master"]):
        if any(x in u for x in ["freediver", "discover", "advanced", "master", "course"]):
            score += 2
    return score


def _text_relevance_score(terms: List[str], text: str) -> int:
    t = text.lower()
    score = 0
    for term in terms:
        if term in t:
            score += 2
            # extra for multiple occurrences
            score += min(3, t.count(term) // 2)
    # If question asks about age/requirements, boost pages mentioning prerequisites
    if any(x in terms for x in ["age", "years", "required", "requirements", "prerequisite"]):
        if any(x in t for x in ["pre-requisites", "prerequisites", "requirements", "minimum age"]):
            score += 4
    return score


def retrieve_site_context(question: str, top_k: int = 4, max_chars: int = 4500) -> List[Dict[str, str]]:
    terms = _query_terms(question)
    urls = get_site_urls()

    if not urls or not terms:
        return []

    # 1) shortlist candidates by URL (no network yet)
    ranked_urls = sorted(urls, key=lambda u: _url_relevance_score(terms, u), reverse=True)
    candidate_urls = ranked_urls[: max(20, MAX_FETCH_PAGES_PER_QUESTION * 3)]

    # 2) fetch only a few pages and score by text
    scored: List[Tuple[int, str, str]] = []
    fetch_budget = MAX_FETCH_PAGES_PER_QUESTION
    for u in candidate_urls:
        if fetch_budget <= 0:
            break
        txt = fetch_page_text(u)
        if not txt:
            continue
        s = _text_relevance_score(terms, txt)
        if s <= 0:
            continue
        scored.append((s, u, txt))
        fetch_budget -= 1

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, str]] = []
    total = 0
    for s, u, txt in scored[:top_k]:
        snippet = txt[:1200]
        chunk = f"URL: {u}\nTEXT: {snippet}\n\n"
        if total + len(chunk) > max_chars:
            break
        out.append({"url": u, "snippet": snippet})
        total += len(chunk)

    return out


# =============================================================================
# Admin Auth
# =============================================================================
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


# =============================================================================
# SendGrid notify (booking / human takeover)
# =============================================================================
def send_owner_email(subject: str, body: str) -> None:
    owner_to = _env("OWNER_NOTIFY_EMAIL")
    api_key = _env("SENDGRID_API_KEY")
    sg_from = _env("SENDGRID_FROM", "SMTP_FROM")
    if not owner_to or not api_key or not sg_from:
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
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code != 202:
            print(f"SendGrid FAILED: status={r.status_code} body={r.text[:500]}")
    except Exception as e:
        print(f"SendGrid FAILED: {type(e).__name__}: {e}")


# =============================================================================
# Calendar (optional)
# =============================================================================
CALENDAR_URL = _env("CALENDAR_URL", default=urljoin(BASE_SITE, "calendar.php"))
_calendar_cache: Dict[str, Any] = {"ts": 0.0, "events": []}


def fetch_calendar_events() -> List[str]:
    now = time.time()
    if now - float(_calendar_cache.get("ts", 0)) < 600 and isinstance(_calendar_cache.get("events"), list):
        return _calendar_cache["events"]

    try:
        r = requests.get(CALENDAR_URL, timeout=HTTP_TIMEOUT)
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


# =============================================================================
# Gemini answering (site first, then web)
# =============================================================================
def try_gemini_answer(question: str, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        return None

    model = _env("GEMINI_MODEL", default="gemini-2.5-flash")

    site_ctx = retrieve_site_context(question, top_k=4)
    have_site = bool(site_ctx)

    web_ctx: List[Dict[str, str]] = []
    if not have_site:
        # web fallback (only if nothing found on site)
        web_ctx = searchapi_web_search(f"site:{BASE_SITE.replace('https://', '').replace('http://', '').strip('/') } {question}", k=5)

    dates = fetch_calendar_events()
    cal_context = (
        "Upcoming dates from the calendar: " + ", ".join(dates)
        if dates
        else "Calendar shows no upcoming dates (confirm availability with the instructor)."
    )

    grounding_text = ""
    if site_ctx:
        grounding_text += "SITE SOURCES:\n" + "\n\n".join(
            [f"URL: {c['url']}\nTEXT: {c['snippet']}" for c in site_ctx]
        )
    if web_ctx:
        grounding_text += "\n\nWEB RESULTS:\n" + "\n\n".join(
            [f"TITLE: {w['title']}\nURL: {w['link']}\nSNIPPET: {w['snippet']}" for w in web_ctx]
        )

    system = (
        "You are Aqua, the assistant for Abood Freediver in Aqaba, Jordan.\n"
        "Be concise, helpful, and safety-first.\n"
        f"{cal_context}\n\n"
        "Rules:\n"
        "- Prefer answering from SITE SOURCES if present.\n"
        "- If SITE SOURCES are empty, you may use WEB RESULTS.\n"
        "- If neither contains the answer, say: \"I couldn’t find this on the website.\" and give a short general answer.\n"
        "- When you use a source, cite the URL(s).\n"
        "- If the user asks to confirm booking/availability, ask for their preferred date/time and tell them you will notify the instructor.\n"
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system + ("\n\n" + grounding_text if grounding_text else "")}]

    if history:
        for m in history[-10:]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": question})

    # New SDK first
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs])

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 650},
        )
        text = getattr(resp, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception:
        pass

    # Old SDK fallback (may be deprecated; keep as backup)
    try:
        import google.generativeai as genai_old  # type: ignore

        genai_old.configure(api_key=api_key)
        gm = genai_old.GenerativeModel(model_name=model, system_instruction=system + ("\n\n" + grounding_text if grounding_text else ""))

        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs if m["role"] != "system"])
        r = gm.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 650},
        )
        text = getattr(r, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception:
        return None

    return None


# =============================================================================
# Minimal fallback (only if Gemini not available)
# =============================================================================
def fallback_answer(question: str) -> Tuple[str, bool]:
    q = question.lower().strip()
    form_url = urljoin(BASE_SITE, "form1.php")

    if any(k in q for k in ["price", "prices", "cost", "how much", "fee"]):
        return (f"Prices are here: {urljoin(BASE_SITE, 'Prices.php?lang=en')}", False)

    if any(k in q for k in ["book", "booking", "reserve", "reservation", "availability", "available", "date", "dates"]):
        return (
            f"To confirm booking/availability, please fill the form: {form_url} "
            "and tell me the date/time you prefer. I will notify the instructor to confirm.",
            True,
        )

    return ("Ask me anything about courses, prices, safety, or availability in Aqaba.", False)


# =============================================================================
# Routes
# =============================================================================
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

    # Only special behavior: booking/availability -> notify owner (after answering)
    booking_like = any(k in q for k in ["book", "booking", "reserve", "reservation", "availability", "available", "date", "dates", "schedule", "time slot"])

    # Try Gemini (site -> web)
    answer = try_gemini_answer(question, req.history)
    if answer:
        needs_human = bool(booking_like)
        if needs_human:
            convo["needs_human"] = True
            send_owner_email(
                subject="Aqua: booking/availability needs confirmation",
                body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nOpen admin:\n/admin",
            )

        convo["messages"].append({"role": "assistant", "text": answer, "ts": datetime.now(timezone.utc).isoformat()})
        return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source="gemini")

    # Fallback (no Gemini key / Gemini error)
    answer2, needs_human2 = fallback_answer(question)
    if needs_human2:
        convo["needs_human"] = True
        send_owner_email(
            subject="Aqua: needs human confirmation",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nOpen admin:\n/admin",
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
