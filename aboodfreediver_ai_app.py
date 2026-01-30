from __future__ import annotations

import os
import re
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import urljoin
import html as html_lib
import secrets
import requests
import xml.etree.ElementTree as ET

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
    source: str = "site"  # "site" | "gemini" | "web" | "fallback"


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


UA_HEADERS = {"User-Agent": "AquaBot/1.0 (+https://www.aboodfreediver.com/)"}


# -----------------------------
# Site/Sitemap indexing (NO need to list all pages in env)
# -----------------------------
BASE_SITE = _env("BASE_SITE_URL", "SITE_BASE_URL", default="https://www.aboodfreediver.com/")
if not BASE_SITE.endswith("/"):
    BASE_SITE += "/"

# You shared this in env (sitemaps.XML) - we accept both cases
SITEMAP_URL = _env("SITE_SITEMAP_URL", "SITEMAP_URL", default=urljoin(BASE_SITE, "sitemaps.xml"))

# Blog index (optional)
BLOG_INDEX_URL = _env("BLOG_SITEMAP_URL", default=urljoin(BASE_SITE, "blog.html"))

# Page cache
_PAGE_CACHE: Dict[str, Dict[str, Any]] = {}  # url -> {"ts": float, "text": str}
_CACHE_TTL_SECONDS = 60 * 60  # 1 hour

# Sitemap cache
_SITEMAP_CACHE: Dict[str, Any] = {"ts": 0.0, "urls": []}
_SITEMAP_TTL_SECONDS = 24 * 60 * 60  # 24h


def _strip_html_to_text(raw_html: str) -> str:
    raw_html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw_html)
    raw_html = re.sub(r"(?is)<[^>]+>", " ", raw_html)
    raw_html = html_lib.unescape(raw_html)
    raw_html = re.sub(r"\s+", " ", raw_html).strip()
    return raw_html


def fetch_page_text(url: str, timeout: int = 15) -> str:
    now = time.time()
    cached = _PAGE_CACHE.get(url)
    if cached and (now - float(cached.get("ts", 0)) < _CACHE_TTL_SECONDS):
        return str(cached.get("text", ""))

    text = ""
    try:
        r = requests.get(url, timeout=timeout, headers=UA_HEADERS, allow_redirects=True)
        r.raise_for_status()
        text = _strip_html_to_text(r.text)
    except Exception:
        text = ""

    _PAGE_CACHE[url] = {"ts": now, "text": text}
    return text


def _safe_get(url: str, timeout: int = 20) -> str:
    r = requests.get(url, timeout=timeout, headers=UA_HEADERS, allow_redirects=True)
    r.raise_for_status()
    return r.text


def _parse_sitemap_xml(xml_text: str) -> List[str]:
    """
    Supports:
      - <urlset> of <url><loc>...</loc>
      - <sitemapindex> of <sitemap><loc>...</loc> (we then fetch child sitemaps)
    """
    out: List[str] = []
    xml_text = xml_text.strip()
    if not xml_text:
        return out

    # Some hosts return HTML error pages; guard:
    if "<html" in xml_text.lower():
        return out

    root = ET.fromstring(xml_text)

    def _localname(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    root_name = _localname(root.tag).lower()

    if root_name == "urlset":
        for child in root:
            if _localname(child.tag).lower() != "url":
                continue
            for item in child:
                if _localname(item.tag).lower() == "loc" and item.text:
                    out.append(item.text.strip())
        return out

    if root_name == "sitemapindex":
        sitemap_locs: List[str] = []
        for child in root:
            if _localname(child.tag).lower() != "sitemap":
                continue
            for item in child:
                if _localname(item.tag).lower() == "loc" and item.text:
                    sitemap_locs.append(item.text.strip())

        # Fetch child sitemaps (cap to avoid abuse)
        for sm in sitemap_locs[:10]:
            try:
                sm_xml = _safe_get(sm, timeout=20)
                out.extend(_parse_sitemap_xml(sm_xml))
            except Exception:
                continue
        return out

    return out


def get_site_urls() -> List[str]:
    """
    Returns a cached list of site URLs discovered from sitemap.
    Falls back to a small list if sitemap can’t be parsed.
    """
    now = time.time()
    if now - float(_SITEMAP_CACHE.get("ts", 0)) < _SITEMAP_TTL_SECONDS and isinstance(_SITEMAP_CACHE.get("urls"), list):
        return list(_SITEMAP_CACHE["urls"])

    urls: List[str] = []
    # Try both cases for your server (sitemaps.XML / sitemaps.xml)
    candidates = list(dict.fromkeys([SITEMAP_URL, SITEMAP_URL.replace(".XML", ".xml"), SITEMAP_URL.replace(".xml", ".XML")]))

    for sm_url in candidates:
        try:
            xml_text = _safe_get(sm_url, timeout=20)
            urls = _parse_sitemap_xml(xml_text)
            if urls:
                break
        except Exception:
            continue

    # Keep only same-site HTML-ish pages
    clean: List[str] = []
    for u in urls:
        if not u.startswith(BASE_SITE):
            continue
        if any(u.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf", ".zip", ".mp4", ".mp3", ".css", ".js", ".ico")):
            continue
        clean.append(u)

    # De-dup
    clean = list(dict.fromkeys(clean))

    if not clean:
        clean = [
            urljoin(BASE_SITE, "index.html"),
            urljoin(BASE_SITE, "FAQ.html"),
            urljoin(BASE_SITE, "Freediver.html"),
            urljoin(BASE_SITE, "Discoverfreediving.html"),
            urljoin(BASE_SITE, "BasicFreediver.html"),
        ]

    _SITEMAP_CACHE["ts"] = now
    _SITEMAP_CACHE["urls"] = clean
    return clean


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


def get_blog_urls(max_posts: int = 30) -> List[str]:
    try:
        r = requests.get(BLOG_INDEX_URL, timeout=15, headers=UA_HEADERS)
        r.raise_for_status()
        links = extract_links(r.text, BLOG_INDEX_URL)
        blog_links = [u for u in links if re.search(r"/blog\d+\.html$", u, re.I)]
        return blog_links[:max_posts]
    except Exception:
        return []


# -----------------------------
# Simple “extractive” QA (no Gemini needed, no “I found links…”)
# -----------------------------
def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 3]


def _split_sentences(text: str) -> List[str]:
    # simple sentence splitter
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) >= 20]


def _score_sentence(q_terms: List[str], sent: str) -> int:
    s = sent.lower()
    score = 0
    for t in set(q_terms):
        if t in s:
            score += 2
    # boost numeric facts
    if re.search(r"\b\d{1,3}\b", sent):
        score += 1
    return score


def _pick_best_sentences(question: str, page_text: str, max_sentences: int = 2) -> str:
    q_terms = _tokenize(question)
    if not q_terms or not page_text:
        return ""

    sents = _split_sentences(page_text)
    if not sents:
        return ""

    scored = [( _score_sentence(q_terms, s), s) for s in sents]
    scored.sort(key=lambda x: x[0], reverse=True)

    best = [s for sc, s in scored if sc > 0][:max_sentences]
    if not best:
        return ""
    # Clean spacing
    answer = " ".join(best)
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def site_extractive_answer(question: str, max_pages_to_fetch: int = 30) -> str:
    """
    1) Use sitemap to get all site URLs
    2) Pick candidate URLs by matching query terms in URL first (fast)
    3) Fetch & extract best sentences from top pages
    """
    urls = get_site_urls()
    q_terms = _tokenize(question)

    # Candidate URLs: term-in-URL heuristic
    def url_score(u: str) -> int:
        ul = u.lower()
        return sum(1 for t in set(q_terms) if t in ul)

    ranked_urls = sorted(urls, key=url_score, reverse=True)

    # If no good URL matches, still try a slice of the site (but cap)
    candidates = ranked_urls[:max_pages_to_fetch]

    best_answer = ""
    best_score = 0

    for u in candidates:
        txt = fetch_page_text(u)
        if not txt:
            continue
        ans = _pick_best_sentences(question, txt, max_sentences=2)
        if not ans:
            continue
        # score answer by term overlap
        score = sum(1 for t in set(q_terms) if t in ans.lower())
        if score > best_score:
            best_score = score
            best_answer = ans
            # early exit if very strong
            if best_score >= max(3, len(set(q_terms)) // 2):
                break

    return best_answer


# -----------------------------
# Web search fallback (only if site couldn’t answer)
# -----------------------------
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
        r = requests.get(url, params=params, timeout=15, headers=UA_HEADERS)
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


def web_snippet_answer(question: str) -> str:
    results = searchapi_web_search(question, k=5)
    if not results:
        return ""
    # Pick best snippet sentences (extractive)
    joined = " ".join([r.get("snippet", "") for r in results if r.get("snippet")])
    return _pick_best_sentences(question, joined, max_sentences=2)


# -----------------------------
# Calendar check (basic scrape + cache)
# -----------------------------
CALENDAR_URL = _env("CALENDAR_URL", default="https://www.aboodfreediver.com/calendar.php")
_calendar_cache: Dict[str, Any] = {"ts": 0.0, "events": []}


def fetch_calendar_events() -> List[str]:
    now = time.time()
    if now - float(_calendar_cache.get("ts", 0)) < 600 and isinstance(_calendar_cache.get("events"), list):
        return list(_calendar_cache["events"])

    try:
        r = requests.get(CALENDAR_URL, timeout=12, headers=UA_HEADERS)
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
# Optional Gemini (ONLY if you want; auto-skips on quota)
# -----------------------------
def try_gemini_answer(question: str, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        return None

    model = _env("GEMINI_MODEL", default="gemini-2.5-flash")

    # Provide a compact site context (best extractive answer + a little text)
    site_ans = site_extractive_answer(question, max_pages_to_fetch=25)
    if not site_ans:
        # still allow gemini, but no site grounding
        site_ans = ""

    dates = fetch_calendar_events()
    cal_context = "Upcoming calendar dates: " + ", ".join(dates) if dates else "Calendar: no listed upcoming dates."

    system = (
        "You are Aqua, the freediving assistant for Abood Freediver in Aqaba, Jordan.\n"
        "Answer directly and clearly.\n"
        "Do NOT say things like 'I found this on these pages' or 'I couldn’t find this'.\n"
        "If uncertain, ask ONE short follow-up question.\n"
        f"\n{cal_context}\n"
        + (f"\nWebsite extracted info:\n{site_ans}\n" if site_ans else "")
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]

    if history:
        for m in history[-8:]:
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
            config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 400},
        )
        text = getattr(resp, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception as e:
        # IMPORTANT: quota errors should not break the app; just skip Gemini
        print(f"Gemini FAILED: {type(e).__name__}: {e}")
        return None

    return None


# -----------------------------
# Minimal fallback
# -----------------------------
def fallback_answer(question: str) -> str:
    # Keep fallback short and neutral (no links dump).
    q = (question or "").strip().lower()

    if any(k in q for k in ["price", "prices", "cost", "how much", "fee"]):
        return "You can see all prices on the Prices page."

    if any(k in q for k in ["book", "booking", "reserve", "reservation"]):
        return "Tell me the date and time you want, and I will confirm availability with the instructor."

    if any(k in q for k in ["whatsapp", "phone", "contact", "email", "number"]):
        return "Use the Contact form or WhatsApp to reach us."

    if any(k in q for k in ["availability", "available", "calendar", "date", "dates", "schedule", "time slot"]):
        dates = fetch_calendar_events()
        if dates:
            return "Next scheduled dates: " + ", ".join(dates) + ". Tell me what day/time you want."
        return "Tell me what day/time you want, and I will confirm availability."

    return "Tell me what you want to do (beginner try, course, fun dive, or training days) and your experience level."


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

    # Human-takeover trigger (booking/availability)
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

    # 1) Try answering from the website directly (fast + reliable + no Gemini quota)
    site_ans = site_extractive_answer(question, max_pages_to_fetch=30)
    if site_ans:
        if needs_human:
            convo["needs_human"] = True
            send_owner_email(
                subject="Aqua: booking/availability confirmation needed",
                body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nCheck /admin to reply.",
            )
        convo["messages"].append({"role": "assistant", "text": site_ans, "ts": datetime.now(timezone.utc).isoformat()})
        return ChatResponse(answer=site_ans, session_id=session_id, needs_human=needs_human, source="site")

    # 2) If site didn't answer, try Gemini (optional; auto-skips on quota)
    gem = try_gemini_answer(question, req.history)
    if gem:
        if needs_human:
            convo["needs_human"] = True
            send_owner_email(
                subject="Aqua: booking/availability confirmation needed",
                body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nCheck /admin to reply.",
            )
        convo["messages"].append({"role": "assistant", "text": gem, "ts": datetime.now(timezone.utc).isoformat()})
        return ChatResponse(answer=gem, session_id=session_id, needs_human=needs_human, source="gemini")

    # 3) If still nothing, try web snippets (SearchApi)
    web_ans = web_snippet_answer(question)
    if web_ans:
        if needs_human:
            convo["needs_human"] = True
            send_owner_email(
                subject="Aqua: booking/availability confirmation needed",
                body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nCheck /admin to reply.",
            )
        convo["messages"].append({"role": "assistant", "text": web_ans, "ts": datetime.now(timezone.utc).isoformat()})
        return ChatResponse(answer=web_ans, session_id=session_id, needs_human=needs_human, source="web")

    # 4) Minimal fallback
    ans = fallback_answer(question)
    if needs_human:
        convo["needs_human"] = True
        send_owner_email(
            subject="Aqua: booking/availability confirmation needed",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nCheck /admin to reply.",
        )
    convo["messages"].append({"role": "assistant", "text": ans, "ts": datetime.now(timezone.utc).isoformat()})
    return ChatResponse(answer=ans, session_id=session_id, needs_human=needs_human, source="fallback")


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
