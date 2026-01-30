from __future__ import annotations

import os
import re
import uuid
import time
import secrets
import requests
import html as html_lib
import xml.etree.ElementTree as ET

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin, urlparse

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
# session_id -> {"messages": [{role, text, ts}], "needs_human": bool, "created": iso}


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
# Config (Site + Search)
# -----------------------------
BASE_SITE = _env("BASE_SITE_URL", "SITE_BASE_URL", default="https://www.aboodfreediver.com/")
if not BASE_SITE.endswith("/"):
    BASE_SITE += "/"

# Prefer sitemap (covers ALL pages without manually listing URLs)
SITEMAP_URL = _env("SITE_SITEMAP_URL", default=urljoin(BASE_SITE, "sitemaps.xml"))

# Blog index is optional; kept for compatibility but not required anymore
BLOG_INDEX_URL = _env("BLOG_SITEMAP_URL", default=urljoin(BASE_SITE, "blog.html"))

# -----------------------------
# Caches
# -----------------------------
_PAGE_CACHE: Dict[str, Dict[str, Any]] = {}  # url -> {"ts": float, "text": str}
_SITEMAP_CACHE: Dict[str, Any] = {"ts": 0.0, "urls": []}
_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours (pages + sitemap)
HTTP = requests.Session()


def fetch_page_text(url: str, timeout: int = 8) -> str:
    now = time.time()
    cached = _PAGE_CACHE.get(url)
    if cached and (now - float(cached.get("ts", 0)) < _CACHE_TTL_SECONDS):
        return str(cached.get("text", ""))

    text = ""
    try:
        r = HTTP.get(url, timeout=timeout, headers={"User-Agent": "AquaBot/1.0"})
        r.raise_for_status()
        text = _strip_html_to_text(r.text)
    except Exception:
        text = ""

    _PAGE_CACHE[url] = {"ts": now, "text": text}
    return text


def _same_site(url: str) -> bool:
    try:
        return urlparse(url).netloc == urlparse(BASE_SITE).netloc
    except Exception:
        return False


def fetch_sitemap_urls(limit: int = 400) -> List[str]:
    """
    Reads sitemap.xml / sitemaps.xml and returns URLs.
    Supports both <urlset> and <sitemapindex> (nested sitemaps).
    """
    now = time.time()
    if now - float(_SITEMAP_CACHE.get("ts", 0)) < _CACHE_TTL_SECONDS and isinstance(_SITEMAP_CACHE.get("urls"), list):
        return list(_SITEMAP_CACHE["urls"])[:limit]

    urls: List[str] = []

    def parse_sitemap(xml_text: str) -> Tuple[List[str], List[str]]:
        locs: List[str] = []
        subs: List[str] = []
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return locs, subs

        tag = root.tag.lower()
        # namespace-safe find
        def iter_loc(parent):
            for el in parent.iter():
                if el.tag.lower().endswith("loc") and (el.text or "").strip():
                    yield (el.text or "").strip()

        # urlset => list of page locs
        if tag.endswith("urlset"):
            for loc in iter_loc(root):
                locs.append(loc)
        # sitemapindex => list of sub-sitemap locs
        elif tag.endswith("sitemapindex"):
            for loc in iter_loc(root):
                subs.append(loc)

        return locs, subs

    def load(url: str) -> str:
        try:
            r = HTTP.get(url, timeout=12, headers={"User-Agent": "AquaBot/1.0"})
            r.raise_for_status()
            return r.text or ""
        except Exception:
            return ""

    main_xml = load(SITEMAP_URL)
    page_locs, sub_sitemaps = parse_sitemap(main_xml)

    if sub_sitemaps:
        # pull a few sub-sitemaps (avoid huge crawling)
        for sm in sub_sitemaps[:10]:
            sm_xml = load(sm)
            locs2, _ = parse_sitemap(sm_xml)
            page_locs.extend(locs2)

    # filter + normalize
    seen = set()
    for u in page_locs:
        if not u:
            continue
        if not _same_site(u):
            continue
        # keep pages likely to contain content (adjust if needed)
        if not re.search(r"\.(html?|php)(\?|#|$)", u, re.I):
            continue
        if u in seen:
            continue
        seen.add(u)
        urls.append(u)

    _SITEMAP_CACHE["ts"] = now
    _SITEMAP_CACHE["urls"] = urls
    return urls[:limit]


def score_text_match(query: str, text: str) -> int:
    # light scoring: count unique terms that appear
    q_terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 3]
    if not q_terms:
        return 0
    t = text.lower()
    return sum(1 for term in set(q_terms) if term in t)


def choose_candidate_urls(question: str, all_urls: List[str], max_candidates: int = 24) -> List[str]:
    """
    Cheap first pass: rank by how many query terms appear in the URL string.
    """
    q_terms = [t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) >= 3]
    if not q_terms:
        return all_urls[:max_candidates]

    scored: List[Tuple[int, str]] = []
    for u in all_urls:
        ul = u.lower()
        s = sum(1 for term in set(q_terms) if term in ul)
        if s > 0:
            scored.append((s, u))

    scored.sort(key=lambda x: x[0], reverse=True)
    if scored:
        return [u for _, u in scored[:max_candidates]]

    # fallback: just take first chunk
    return all_urls[:max_candidates]


def retrieve_site_context(question: str, top_k: int = 5, max_chars: int = 5500) -> List[Dict[str, str]]:
    all_urls = fetch_sitemap_urls(limit=400)
    candidates = choose_candidate_urls(question, all_urls, max_candidates=24)

    # Second pass: fetch a small number of candidate pages and score by content
    scored: List[Tuple[int, str, str]] = []
    for u in candidates[:10]:  # keep it fast (Render free tier)
        txt = fetch_page_text(u, timeout=8)
        if not txt:
            continue
        s = score_text_match(question, txt)
        if s <= 0:
            continue
        scored.append((s, u, txt))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, str]] = []
    total = 0
    for s, u, txt in scored[:top_k]:
        snippet = txt[:1200]
        chunk_len = len(u) + len(snippet) + 20
        if total + chunk_len > max_chars:
            break
        out.append({"url": u, "snippet": snippet})
        total += chunk_len

    return out


def searchapi_web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    api_key = _env("SEARCHAPI_KEY")
    if not api_key:
        return []
    try:
        url = "https://www.searchapi.io/api/v1/search"
        params = {"engine": "google", "q": query, "api_key": api_key, "hl": "en", "gl": "us", "page": 1}
        r = HTTP.get(url, params=params, timeout=15)
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


# -----------------------------
# Calendar scrape (optional)
# -----------------------------
CALENDAR_URL = _env("CALENDAR_URL", default=urljoin(BASE_SITE, "calendar.php"))
_calendar_cache: Dict[str, Any] = {"ts": 0.0, "events": []}


def fetch_calendar_events() -> List[str]:
    now = time.time()
    if now - float(_calendar_cache.get("ts", 0)) < 600 and isinstance(_calendar_cache.get("events"), list):
        return list(_calendar_cache["events"])

    try:
        r = HTTP.get(CALENDAR_URL, timeout=10, headers={"User-Agent": "AquaBot/1.0"})
        r.raise_for_status()
        html = r.text or ""
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
# Email notify via SendGrid API (Render-safe: HTTPS)
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
        r = HTTP.post(
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
# Gemini answering (Site -> Web fallback)
# -----------------------------
def try_gemini_answer(question: str, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        return None

    model = _env("GEMINI_MODEL", default="gemini-2.5-flash")

    # 1) Search the whole website via sitemap
    site_ctx = retrieve_site_context(question, top_k=5)

    # 2) If nothing found on site, search the web
    web_ctx: List[Dict[str, str]] = []
    if not site_ctx:
        web_ctx = searchapi_web_search(f"site:{urlparse(BASE_SITE).netloc} {question}", k=5)
        if not web_ctx:
            web_ctx = searchapi_web_search(question, k=5)

    dates = fetch_calendar_events()
    cal_context = (
        "Calendar upcoming dates: " + ", ".join(dates)
        if dates
        else "Calendar: no upcoming dates detected (availability must be confirmed)."
    )

    grounding_text = ""
    if site_ctx:
        grounding_text += "WEBSITE PAGES:\n" + "\n\n".join(
            [f"URL: {c['url']}\nTEXT: {c['snippet']}" for c in site_ctx]
        )
    if web_ctx:
        grounding_text += "\n\nWEB RESULTS:\n" + "\n\n".join(
            [f"TITLE: {w['title']}\nURL: {w['link']}\nSNIPPET: {w['snippet']}" for w in web_ctx]
        )

    # Minimal instructions (as requested)
    system = (
        "You are Aqua, the assistant for Abood Freediver in Aqaba, Jordan.\n"
        "Answer using the provided WEBSITE PAGES first. If not present there, use WEB RESULTS.\n"
        "If the answer is not supported by those sources, say: \"I couldn’t find this on the website.\" "
        "Then give a short general answer.\n"
        "When you use information from sources, include the URL(s).\n\n"
        f"{cal_context}\n\n"
        f"{grounding_text}".strip()
    )

    # Build a single prompt (works reliably across environments)
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]
    if history:
        for m in history[-10:]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": question})

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs])

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 700},
        )
        text = getattr(resp, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception as e:
        print(f"Gemini FAILED: {type(e).__name__}: {e}")

    return None


# -----------------------------
# Fallback (no Gemini key): Site -> Web -> generic
# -----------------------------
def fallback_answer(question: str) -> str:
    site_ctx = retrieve_site_context(question, top_k=3)
    if site_ctx:
        # very simple summary without LLM
        urls = ", ".join([c["url"] for c in site_ctx])
        return f"I found relevant info on these pages: {urls}\n\nTry opening the page(s) above for the exact details."

    web_ctx = searchapi_web_search(f"site:{urlparse(BASE_SITE).netloc} {question}", k=3)
    if web_ctx:
        urls = ", ".join([w["link"] for w in web_ctx if w.get("link")])
        return f"I couldn’t find this on the website. Here are relevant results: {urls}"

    return "I couldn’t find this on the website. Please use the contact form: https://www.aboodfreediver.com/form1.php"


def needs_human_confirmation(question: str) -> bool:
    q = (question or "").lower()
    # only for booking / availability confirmation (minimal rules)
    triggers = [
        "book",
        "booking",
        "reserve",
        "reservation",
        "availability",
        "available",
        "calendar",
        "date",
        "dates",
        "schedule",
        "time slot",
    ]
    return any(t in q for t in triggers)


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

    # 1) Gemini (site -> web) if available
    answer = try_gemini_answer(question, req.history)
    source = "gemini" if answer else "fallback"

    # 2) Fallback if Gemini unavailable / failed
    if not answer:
        answer = fallback_answer(question)

    # 3) Human confirmation flow (only for booking/availability)
    needs_human = needs_human_confirmation(question)
    if needs_human:
        convo["needs_human"] = True
        send_owner_email(
            subject="Aqua needs confirmation (booking/availability)",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nOpen admin:\n/admin",
        )

    convo["messages"].append({"role": "assistant", "text": answer, "ts": datetime.now(timezone.utc).isoformat()})
    return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source=source)


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
