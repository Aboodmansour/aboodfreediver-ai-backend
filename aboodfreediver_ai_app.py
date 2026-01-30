from __future__ import annotations

import os
import re
import uuid
import time
import asyncio
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import secrets
import requests
from urllib.parse import urljoin
import html as html_lib

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
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
# Background warming on startup (best performance)
# -----------------------------
async def _warm_cache_and_index() -> None:
    try:
        # warm service urls
        svc_urls = get_service_urls(max_urls=60)
        # warm key pages quickly in threads
        async def warm_url(u: str) -> None:
            await asyncio.to_thread(fetch_page_text, u, 12)

        tasks = [warm_url(u) for u in svc_urls[:60]]
        await asyncio.gather(*tasks, return_exceptions=True)

        # warm some blog pages
        blog_urls = get_blog_urls(max_posts=25)
        tasks2 = [warm_url(u) for u in blog_urls[:25]]
        await asyncio.gather(*tasks2, return_exceptions=True)

        # build BM25 index using warmed service urls (best coverage)
        await asyncio.to_thread(_build_bm25_index, svc_urls[:80], "services")
        _log("Warm cache done. Indexed docs:", int(_DOC_INDEX.get("n") or 0))
    except Exception as e:
        _log("Warm cache failed:", type(e).__name__, str(e))

@app.on_event("startup")
async def _startup_event():
    # fire-and-forget warming task; does not block requests
    asyncio.create_task(_warm_cache_and_index())



# -----------------------------
# In-memory storage (Render free tier resets on deploy/sleep)
# -----------------------------
CONVERSATIONS: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# Realtime hub (in-memory pub/sub per session)
# -----------------------------
# session_id -> set of asyncio.Queue for user/admin listeners
_SESSION_LISTENERS: Dict[str, List[asyncio.Queue]] = {}

# admin auth tokens (short-lived) for websocket access
_ADMIN_WS_TOKENS: Dict[str, Dict[str, Any]] = {}  # token -> {"exp": float}

def _ws_token_new(ttl_seconds: int = 1800) -> str:
    token = secrets.token_urlsafe(32)
    _ADMIN_WS_TOKENS[token] = {"exp": time.time() + ttl_seconds}
    return token

def _ws_token_valid(token: str) -> bool:
    data = _ADMIN_WS_TOKENS.get(token)
    if not data:
        return False
    if time.time() > float(data.get("exp", 0)):
        _ADMIN_WS_TOKENS.pop(token, None)
        return False
    return True

def _listeners_for(session_id: str) -> List[asyncio.Queue]:
    return _SESSION_LISTENERS.setdefault(session_id, [])

def _listener_add(session_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _listeners_for(session_id).append(q)
    return q

def _listener_remove(session_id: str, q: asyncio.Queue) -> None:
    listeners = _SESSION_LISTENERS.get(session_id) or []
    try:
        listeners.remove(q)
    except ValueError:
        pass
    if not listeners:
        _SESSION_LISTENERS.pop(session_id, None)

def _broadcast(session_id: str, payload: Dict[str, Any]) -> None:
    listeners = list(_SESSION_LISTENERS.get(session_id) or [])
    for q in listeners:
        try:
            q.put_nowait(payload)
        except Exception:
            # if full or closed, ignore; clients can reconnect
            pass

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


DEBUG = _env("DEBUG", default="0").strip() in ("1", "true", "True", "YES", "yes")


def _log(*args: Any) -> None:
    if DEBUG:
        print("[AQUA]", *args)


# -----------------------------
# Site/Blog Retrieval Helpers (site -> blog -> web)
# -----------------------------
BASE_SITE = _env("BASE_SITE_URL", "SITE_BASE_URL", default="https://www.aboodfreediver.com/")
if not BASE_SITE.endswith("/"):
    BASE_SITE += "/"

# Your env uses BLOG_SITEMAP_URL and it points to blog.html (works as blog index)
BLOG_INDEX_URL = _env("BLOG_SITEMAP_URL", default=urljoin(BASE_SITE, "blog.html"))

_PAGE_CACHE: Dict[str, Dict[str, Any]] = {}  # url -> {"ts": float, "text": str}
_CACHE_TTL_SECONDS = 60 * 60  # 1 hour

# -----------------------------
# Lightweight search index (BM25; no embeddings)
# -----------------------------
_DOC_INDEX: Dict[str, Any] = {
    "ts": 0.0,
    "docs": [],      # list[{"url": str, "text": str, "tf": dict[str,int], "len": int}]
    "idf": {},       # dict[str,float]
    "avgdl": 0.0,
    "n": 0,
    "source": "none",
}

def _tokenize(text: str) -> List[str]:
    # keep letters+digits across languages; Arabic letters are kept by \w in unicode mode
    tokens = re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
    return [t for t in tokens if len(t) >= 2]

def _build_bm25_index(urls: List[str], source: str) -> None:
    docs = []
    df: Dict[str, int] = {}
    total_len = 0

    for u in urls:
        txt = fetch_page_text(u)
        if not txt:
            continue
        tokens = _tokenize(txt)
        if not tokens:
            continue
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t in set(tf.keys()):
            df[t] = df.get(t, 0) + 1
        dl = len(tokens)
        total_len += dl
        docs.append({"url": u, "text": txt, "tf": tf, "len": dl})

    n = len(docs)
    if n == 0:
        _DOC_INDEX.update({"ts": time.time(), "docs": [], "idf": {}, "avgdl": 0.0, "n": 0, "source": source})
        return

    avgdl = total_len / n
    idf: Dict[str, float] = {}
    for term, dfi in df.items():
        # BM25 idf with +1 to avoid negatives
        idf[term] = math.log(1 + (n - dfi + 0.5) / (dfi + 0.5))

    _DOC_INDEX.update({"ts": time.time(), "docs": docs, "idf": idf, "avgdl": avgdl, "n": n, "source": source})

def _bm25_score(query: str, doc: Dict[str, Any], k1: float = 1.5, b: float = 0.75) -> float:
    q_terms = _tokenize(query)
    if not q_terms:
        return 0.0
    tf = doc["tf"]
    dl = float(doc["len"])
    avgdl = float(_DOC_INDEX.get("avgdl") or 0.0) or 1.0
    score = 0.0
    for t in q_terms:
        if t not in tf:
            continue
        f = float(tf[t])
        idf = float(_DOC_INDEX["idf"].get(t, 0.0))
        denom = f + k1 * (1 - b + b * (dl / avgdl))
        score += idf * (f * (k1 + 1) / (denom if denom else 1.0))
    return score



def searchapi_web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """
    Uses SearchApi.io Google Search API.
    Env: SEARCHAPI_KEY
    """
    api_key = _env("SEARCHAPI_KEY")
    if not api_key:
        _log("SEARCHAPI_KEY missing -> web search disabled")
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
        r = requests.get(url, params=params, timeout=15)
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
        _log("WEB_SEARCH results:", len(out))
        return out
    except Exception as e:
        _log("WEB_SEARCH failed:", type(e).__name__, str(e))
        return []


def _strip_html_to_text(raw_html: str) -> str:
    """Lightweight HTML->text without extra deps."""
    raw_html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw_html)
    raw_html = re.sub(r"(?is)<[^>]+>", " ", raw_html)
    raw_html = html_lib.unescape(raw_html)
    raw_html = re.sub(r"\s+", " ", raw_html).strip()
    return raw_html


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


def get_main_site_urls() -> List[str]:
    return [
        urljoin(BASE_SITE, "index.html"),
        urljoin(BASE_SITE, "FAQ.html"),
        urljoin(BASE_SITE, "Discoverfreediving.html"),
        urljoin(BASE_SITE, "BasicFreediver.html"),
        urljoin(BASE_SITE, "Freediver.html"),
        urljoin(BASE_SITE, "Advancedfreediver.html"),
        urljoin(BASE_SITE, "trainingsession.html"),
        urljoin(BASE_SITE, "FunFreediving.html"),
        urljoin(BASE_SITE, "snorkelguide.html"),
        urljoin(BASE_SITE, "freedivingequipment.html"),
        urljoin(BASE_SITE, "divesites.html"),
        urljoin(BASE_SITE, "cedar-pride.html"),
        urljoin(BASE_SITE, "military.html"),
        urljoin(BASE_SITE, "Tristar.html"),
        urljoin(BASE_SITE, "C-130.html"),
        urljoin(BASE_SITE, "Tank.html"),
    ]

COURSE_URLS: List[str] = [
    "https://www.aboodfreediver.com/Discoverfreediving.html",
    "https://www.aboodfreediver.com/BasicFreediver.html",
    "https://www.aboodfreediver.com/Freediver.html",
    "https://www.aboodfreediver.com/trainingsession.html",
    "https://www.aboodfreediver.com/FunFreediving.html",
    "https://www.aboodfreediver.com/Advancedfreediver.html",
    "https://www.aboodfreediver.com/snorkelguide.html",
]


# -----------------------------
# Service page discovery (crawl internal links from key pages)
# -----------------------------
_SERVICE_URLS_CACHE: Dict[str, Any] = {"ts": 0.0, "urls": []}
_SERVICE_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours

def get_service_urls(max_urls: int = 60) -> List[str]:
    """Discover additional internal service pages so Aqua can answer without keyword rules."""
    now = time.time()
    if now - float(_SERVICE_URLS_CACHE.get("ts", 0)) < _SERVICE_CACHE_TTL_SECONDS and _SERVICE_URLS_CACHE.get("urls"):
        return list(_SERVICE_URLS_CACHE["urls"])

    seed_pages = [
        urljoin(BASE_SITE, "index.html"),
        urljoin(BASE_SITE, "FAQ.html"),
        urljoin(BASE_SITE, "Discoverfreediving.html"),
        urljoin(BASE_SITE, "BasicFreediver.html"),
        urljoin(BASE_SITE, "Freediver.html"),
        urljoin(BASE_SITE, "Advancedfreediver.html"),
        urljoin(BASE_SITE, "trainingsession.html"),
        urljoin(BASE_SITE, "FunFreediving.html"),
        urljoin(BASE_SITE, "snorkelguide.html"),
    ]

    urls: List[str] = []
    for seed in seed_pages:
        try:
            r = requests.get(seed, timeout=12, headers={"User-Agent": "AquaBot/1.0"})
            r.raise_for_status()
            urls.extend(extract_links(r.text, seed))
        except Exception:
            continue

    cleaned: List[str] = []
    for link in urls:
        if not link.startswith(BASE_SITE):
            continue
        if re.search(r"\.(html|php)$", link, re.I):
            cleaned.append(link)

    cleaned = list(dict.fromkeys(cleaned))
    merged = list(dict.fromkeys(get_main_site_urls() + cleaned))[:max_urls]

    _SERVICE_URLS_CACHE["ts"] = now
    _SERVICE_URLS_CACHE["urls"] = merged
    return merged


def get_blog_urls(max_posts: int = 30) -> List[str]:
    try:
        r = requests.get(BLOG_INDEX_URL, timeout=12, headers={"User-Agent": "AquaBot/1.0"})
        r.raise_for_status()
        links = extract_links(r.text, BLOG_INDEX_URL)
        # your blog looks like blog1.html, blog2.html, ...
        blog_links = [u for u in links if re.search(r"/blog\d+\.html$", u, re.I)]
        if blog_links:
            return blog_links[:max_posts]
        # fallback if blog index doesn't list them clearly
        return [urljoin(BASE_SITE, f"blog{i}.html") for i in range(1, 21)]
    except Exception:
        return [urljoin(BASE_SITE, f"blog{i}.html") for i in range(1, 21)]


def score_text_match(query: str, text: str) -> int:
    q_terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 3]
    if not q_terms:
        return 0
    t = text.lower()
    score = 0
    for term in set(q_terms):
        if term in t:
            score += 1
    return score


def retrieve_site_context(
    question: str,
    urls: List[str],
    top_k: int = 4,
    max_chars: int = 4000,
    time_budget: float = 2.5,
) -> List[Dict[str, str]]:
    """Return top matching page snippets with a strict time budget (prevents request timeouts)."""
    start = time.time()

    # If we have a BM25 index, use it (fast). Otherwise fallback to lightweight scoring on fetched texts.
    use_index = bool(_DOC_INDEX.get("docs")) and time.time() - float(_DOC_INDEX.get("ts", 0)) < 8 * 60 * 60
    scored: List[Tuple[float, str, str]] = []

    if use_index:
        for d in _DOC_INDEX["docs"]:
            if time.time() - start > time_budget:
                break
            s = _bm25_score(question, d)
            if s <= 0:
                continue
            scored.append((s, d["url"], d["text"]))
    else:
        for u in urls:
            if time.time() - start > time_budget:
                break
            txt = fetch_page_text(u)
            if not txt:
                continue
            s = float(score_text_match(question, txt))
            if s <= 0:
                continue
            scored.append((s, u, txt))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, str]] = []
    total = 0
    for s, u, txt in scored[:top_k]:
        snippet = txt[: min(len(txt), 1400)]
        chunk = f"URL: {u}\nTEXT: {snippet}\n\n"
        if total + len(chunk) > max_chars:
            break
        out.append({"url": u, "snippet": snippet})
        total += len(chunk)

    _log("SITE_CTX pages:", len(out))
    return out

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
CALENDAR_URL = _env("CALENDAR_URL", default=urljoin(BASE_SITE, "calendar.php"))
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
        r = requests.get(CALENDAR_URL, timeout=10, headers={"User-Agent": "AquaBot/1.0"})
        r.raise_for_status()
        page_html = r.text

        date_regex = re.compile(
            r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b",
            re.IGNORECASE,
        )
        dates = list(dict.fromkeys(date_regex.findall(page_html)))[:5]

        _calendar_cache["ts"] = now
        _calendar_cache["events"] = dates
        return dates
    except Exception:
        _calendar_cache["ts"] = now
        _calendar_cache["events"] = []
        return []


# -----------------------------
# Gemini (site -> blog -> web; keeps booking/email logic unchanged)
# -----------------------------
def try_gemini_answer(question: str, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        _log("GEMINI key missing -> gemini disabled")
        return None

    model = _env("GEMINI_MODEL", default="gemini-2.5-flash")

    lang = detect_lang(question)

    # 1) Retrieve context: main site -> blog
    # Course pages are always checked first for course-related questions
    course_ctx: List[Dict[str, str]] = []
    if any(k in question.lower() for k in ["course", "courses", "level", "levels", "discover", "basic", "freediver", "advanced", "training", "session", "snorkel"]):
        course_ctx = retrieve_site_context(question, COURSE_URLS, top_k=4, max_chars=3000, time_budget=1.2)
    main_ctx = retrieve_site_context(question, get_service_urls())
    blog_ctx: List[Dict[str, str]] = []
    if not main_ctx:
        blog_ctx = retrieve_site_context(question, get_blog_urls())

    have_grounding = bool(course_ctx or main_ctx or blog_ctx)

    # Confidence is based on how strong the site/blog match is.
    best_score = 0.0
    try:
        if _DOC_INDEX.get('docs'):
            # approximate: max bm25 across docs for this query within a small budget
            start = time.time()
            for d in _DOC_INDEX['docs'][:120]:
                if time.time() - start > 0.25:
                    break
                best_score = max(best_score, _bm25_score(question, d))
    except Exception:
        pass
    # scale to 0..1 (heuristic)
    confidence = max(0.0, min(1.0, best_score / 8.0))

    # 2) Web search only if site/blog didn't match
    web_ctx: List[Dict[str, str]] = []
    external_agency = any(k in question.lower() for k in ["padi", "aida", "molchanovs", "ssi"]) 
    if ((not have_grounding) and confidence < 0.35) or external_agency:
        web_ctx = searchapi_web_search(question, k=5)

    dates = fetch_calendar_events()
    cal_context = (
        "Upcoming dates from the calendar: " + ", ".join(dates)
        if dates
        else "Calendar shows no upcoming dates (may mean mostly free; must confirm with instructor)."
    )

    grounding_text = ""
    if course_ctx:
        grounding_text += "COURSE SOURCES:\n" + "\n\n".join(
            [f"URL: {c['url']}\nTEXT: {c['snippet']}" for c in course_ctx]
        )
    if main_ctx:
        grounding_text += "MAIN SITE SOURCES:\n" + "\n\n".join(
            [f"URL: {c['url']}\nTEXT: {c['snippet']}" for c in main_ctx]
        )
    if blog_ctx:
        grounding_text += "\n\nBLOG SOURCES:\n" + "\n\n".join(
            [f"URL: {c['url']}\nTEXT: {c['snippet']}" for c in blog_ctx]
        )
    if web_ctx:
        grounding_text += "\n\nWEB RESULTS:\n" + "\n\n".join(
            [f"TITLE: {w['title']}\nURL: {w['link']}\nSNIPPET: {w['snippet']}" for w in web_ctx]
        )

    system = (
        "You are Aqua, the freediving assistant for Abood Freediver in Aqaba, Jordan (Red Sea).\nYou ONLY answer questions about freediving, freediving training/safety, and Abood Freediver services.\nIf the user asks about topics unrelated to freediving/Abood Freediver, say you can’t help with that and ask them to rephrase a freediving-related question.\n"
        "Always prioritize safety. If the user asks for medical advice, recommend seeing a professional.\n"
        "Respond in the same language as the user (Arabic if they write Arabic, otherwise English).\n\n"
        "Hard business rules (the ONLY thing that requires human confirmation is booking/availability):\n"
        "1) If asked about prices: answer with the exact prices if they appear in MAIN SITE / SERVICES sources; also include this link for reference: https://www.aboodfreediver.com/Prices.php?lang=en\n"
        "2) If asked about contact/booking, link to: https://www.aboodfreediver.com/form1.php\n"
        "3) If asked about availability/dates (requires instructor confirmation):\n"
        "   - If calendar has events, mention the next dates.\n"
        "   - If calendar has no events, say we are usually free BUT must confirm with the instructor.\n\n"
        "Content rules (IMPORTANT):\n"
        "- Use ONLY the provided MAIN SITE / BLOG SOURCES and (if present) WEB RESULTS for factual claims.\n"
        "- If the answer is not in those sources, say exactly: \"\" "
        "Then provide a short best-effort general answer.\n"
        "- When you use info from sources, cite the URL(s) in the answer.\n"
    )

    msgs: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": system + "\n\n" + cal_context + ("\n\n" + grounding_text if grounding_text else "") + f"\n\nCONFIDENCE_HINT: {confidence:.2f}",
        }
    ]

    if history:
        for m in history[-12:]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": question})

    
def gemini_rest_generate(api_key: str, model: str, prompt: str) -> Optional[str]:
    """Direct REST call to Gemini API (avoids SDK issues)."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        r = requests.post(
            url,
            params={"key": api_key},
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "topP": 0.9, "maxOutputTokens": 700},
            },
            timeout=25,
        )
        r.raise_for_status()
        data = r.json() or {}
        cands = data.get("candidates") or []
        if not cands:
            return None
        parts = (((cands[0] or {}).get("content") or {}).get("parts") or [])
        text = " ".join([str(p.get("text", "")).strip() for p in parts if isinstance(p, dict)]).strip()
        return text or None
    except Exception as e:
        _log("gemini REST failed:", type(e).__name__, str(e))
        return None

# --- Try new SDK first: google-genai ---
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
        _log("google-genai failed:", type(e).__name__, str(e))

    # --- Fallback old SDK: google-generativeai ---
    try:
        import google.generativeai as genai_old  # type: ignore

        genai_old.configure(api_key=api_key)
        gm = genai_old.GenerativeModel(
            model_name=model,
            system_instruction=system + "\n\n" + cal_context + ("\n\n" + grounding_text if grounding_text else ""),
        )

        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs if m["role"] != "system"])
        r = gm.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 700},
        )
        text = getattr(r, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception as e:
        _log("google-generativeai failed:", type(e).__name__, str(e))
        return None

    return None



def detect_lang(text: str) -> str:
    # very simple Arabic detection
    if re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text):
        return "ar"
    return "en"


def is_freediving_related(question: str) -> bool:
    q = question.lower()
    # allow common business interactions
    if any(k in q for k in ["price", "prices", "cost", "how much", "fee",
                           "book", "booking", "reserve", "reservation",
                           "availability", "available", "calendar", "date", "dates", "schedule", "time slot",
                           "whatsapp", "phone", "contact", "email", "number",
                           "open", "opening", "hours", "working hours",
                           "aqaba", "abood", "red sea", "jordan"]):
        return True

    # core freediving + diving topics
    keywords = [
        "freediv", "free div", "apnea", "breath hold", "breathhold",
        "equaliz", "frenzel", "valsalva", "mask", "fins", "monofin",
        "wetsuit", "neck weight", "lanyard", "buoy", "float", "line",
        "depth", "dynamic", "static", "cwt", "cnf", "fimt", "no fins",
        "safety", "rescue", "blackout", "samba", "hypoxia",
        "course", "cert", "certification", "aida", "molchanovs", "ssi",
        "training", "coach", "instructor", "session", "dive site", "divesite",
        "equipment", "weight", "weights", "ear", "sinus", "pressure",
    ]
    return any(k in q for k in keywords)


# -----------------------------
# Fallback logic
# -----------------------------
def fallback_answer(question: str) -> Tuple[str, bool]:
    q = question.strip().lower()
    lang = detect_lang(question)
    FORM_URL = "https://www.aboodfreediver.com/form1.php"

    # Minimal safe fallback if Gemini is unavailable
    if any(k in q for k in ["book", "booking", "reserve", "reservation", "availability", "available", "date", "dates", "schedule"]):
        return ((f"Please share your preferred date/time and I will confirm availability. Booking form: {FORM_URL}") if lang=="en" else (f"من فضلك أخبرني بالتاريخ/الوقت الذي تريده وسأؤكد التوفر. نموذج الحجز: {FORM_URL}"), True)

    return (("I can help with freediving training, safety, equipment, courses, and Abood Freediver services in Aqaba. What would you like to know?") if lang=="en" else "أستطيع مساعدتك في تدريب الغوص الحر، السلامة، المعدات، الدورات، وخدمات Abood Freediver في العقبة. ماذا تريد أن تعرف؟", False)

# -----------------------------
# Routes
# -----------------------------
@app.head("/")
def root_head():
    return {"ok": True}


@app.get("/")
def root():
    # avoids Render showing lots of 404s for GET /
    return {"ok": True, "service": "Aqua", "time": datetime.now(timezone.utc).isoformat()}


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
    _broadcast(session_id, {"role": "user", "text": question, "ts": datetime.now(timezone.utc).isoformat()})

    q = question.lower().strip()
    FORM_URL = "https://www.aboodfreediver.com/form1.php"

    # RULE OVERRIDES removed: Aqua now searches sources instead of keyword answers.

    
    # Proactive clarifying questions (reduces long/uncertain answers)
    qlow = q
    clarify_terms = ["when", "next", "which course", "what course", "recommend", "best course", "schedule", "available", "availability", "dates", "date"]
    if any(t in qlow for t in clarify_terms):
        # if user asks "when next course" but doesn't specify dates
        if ("when" in qlow or "next" in qlow) and not re.search(r"\b\d{1,2}[/-]\d{1,2}\b|\b\d{4}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", qlow):
            # let Gemini answer briefly, but prefer a clarifying question in the response context
            pass
# try gemini (site -> blog -> web)
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
        _broadcast(session_id, {"role": "assistant", "text": answer, "ts": datetime.now(timezone.utc).isoformat(), "needs_human": needs_human})
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
    _broadcast(session_id, {"role": "assistant", "text": answer2, "ts": datetime.now(timezone.utc).isoformat(), "needs_human": needs_human2})
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
    _broadcast(data.session_id, {"role": "human", "text": data.message, "ts": datetime.now(timezone.utc).isoformat(), "needs_human": False})
    convo["needs_human"] = False
    return {"ok": True}



@app.get("/admin/token")
def admin_ws_token(_: bool = Depends(admin_auth)):
    # short-lived token used for websocket auth
    return {"token": _ws_token_new(ttl_seconds=1800)}


@app.websocket("/ws/user/{session_id}")
async def ws_user(session_id: str, ws: WebSocket):
    await ws.accept()
    q = _listener_add(session_id)
    try:
        # send current history snapshot
        convo = CONVERSATIONS.get(session_id) or {"messages": [], "needs_human": False}
        await ws.send_json({"type": "snapshot", "session_id": session_id, "needs_human": convo.get("needs_human", False), "messages": convo.get("messages", [])})

        while True:
            payload = await q.get()
            await ws.send_json({"type": "event", "session_id": session_id, **payload})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _listener_remove(session_id, q)


@app.websocket("/ws/admin/{session_id}")
async def ws_admin(session_id: str, ws: WebSocket, token: str = Query(default="")):
    # token is required for admin websocket
    if not token or not _ws_token_valid(token):
        # 1008 = policy violation
        await ws.close(code=1008)
        return

    await ws.accept()
    q = _listener_add(session_id)
    try:
        convo = CONVERSATIONS.get(session_id) or {"messages": [], "needs_human": False}
        await ws.send_json({"type": "snapshot", "session_id": session_id, "needs_human": convo.get("needs_human", False), "messages": convo.get("messages", [])})

        while True:
            data = await ws.receive_json()
            msg = str((data or {}).get("message", "")).strip()
            if not msg:
                continue

            convo2 = CONVERSATIONS.get(session_id)
            if not convo2:
                # create if admin starts first
                convo2 = CONVERSATIONS.setdefault(
                    session_id,
                    {"messages": [], "needs_human": False, "created": datetime.now(timezone.utc).isoformat()},
                )

            convo2["messages"].append({"role": "human", "text": msg, "ts": datetime.now(timezone.utc).isoformat()})
            convo2["needs_human"] = False
            _broadcast(session_id, {"role": "human", "text": msg, "ts": datetime.now(timezone.utc).isoformat(), "needs_human": False})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _listener_remove(session_id, q)


@app.get("/admin", response_class=HTMLResponse)
def admin_page(_: bool = Depends(admin_auth)):
    return """
<!DOCTYPE html>
<html>
<head>
  <title>Aqua – Instructor Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: Arial; padding: 20px; max-width: 980px; margin: 0 auto; }
    input, textarea, button { width: 100%; margin: 10px 0; padding: 10px; font-size: 16px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; }
    .chat { height: 520px; overflow: auto; background: #fafafa; border: 1px solid #eee; border-radius: 8px; padding: 10px; }
    .msg { margin: 8px 0; padding: 8px 10px; border-radius: 10px; max-width: 90%; white-space: pre-wrap; }
    .user { background: #ffffff; border: 1px solid #eee; }
    .assistant { background: #eef7ff; border: 1px solid #d7ecff; }
    .human { background: #f5fff1; border: 1px solid #e2ffd7; }
    .meta { font-size: 12px; color: #555; margin-bottom: 2px; }
    .badge { display: inline-block; padding: 4px 8px; border-radius: 999px; background: #eee; font-size: 12px; }
    .needs { background: #ffe8e8; }
    .ok { background: #e9ffe8; }
  </style>
</head>
<body>
  <h2>Aqua – Admin Real-time Chat</h2>
  <div class="card">
    <div class="meta">This page uses HTTP Basic Auth. Enter the session_id from the user chat response.</div>
    <input id="sid" placeholder="Session ID" />
    <button onclick="connectWs()">Connect (real-time)</button>
    <div id="status" class="badge">Disconnected</div>
  </div>

  <div class="row">
    <div class="card">
      <h3>Conversation</h3>
      <div id="chat" class="chat"></div>
    </div>

    <div class="card">
      <h3>Reply as Human Instructor</h3>
      <textarea id="msg" rows="4" placeholder="Type your reply..."></textarea>
      <button onclick="sendHuman()">Send</button>

      <h3>Manual tools</h3>
      <button onclick="loadStatus()">Load chat status (HTTP)</button>
      <pre id="out"></pre>
    </div>
  </div>

<script>
let ws = null;
let wsToken = null;

function esc(s) {
  return (s || "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderMessage(m) {
  const role = m.role || 'unknown';
  const text = m.text || '';
  const ts = m.ts || '';
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  wrap.innerHTML = `<div class="meta">${esc(role)} • ${esc(ts)}</div>${esc(text)}`;
  return wrap;
}

function setStatus(text, needsHuman=false) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = 'badge ' + (needsHuman ? 'needs' : 'ok');
}

async function getToken() {
  // requires basic auth already accepted by browser
  const r = await fetch('/admin/token');
  if (!r.ok) throw new Error('Failed to get token: ' + r.status);
  const j = await r.json();
  return j.token;
}

async function connectWs() {
  const sid = document.getElementById('sid').value.trim();
  if (!sid) return alert('Session ID is required');

  if (ws) { try { ws.close(); } catch(e) {} ws = null; }

  setStatus('Connecting...', false);

  try {
    wsToken = await getToken();
  } catch (e) {
    setStatus('Token error', true);
    document.getElementById('out').textContent = String(e);
    return;
  }

  const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/admin/${encodeURIComponent(sid)}?token=${encodeURIComponent(wsToken)}`);

  ws.onopen = () => setStatus('Connected', false);

  ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      if (data.type === 'snapshot') {
        const chat = document.getElementById('chat');
        chat.innerHTML = '';
        (data.messages || []).forEach(m => chat.appendChild(renderMessage(m)));
        chat.scrollTop = chat.scrollHeight;
        setStatus('Connected', !!data.needs_human);
      } else if (data.type === 'event') {
        const chat = document.getElementById('chat');
        chat.appendChild(renderMessage(data));
        chat.scrollTop = chat.scrollHeight;
        if (typeof data.needs_human !== 'undefined') {
          setStatus('Connected', !!data.needs_human);
        }
      }
    } catch (e) {
      document.getElementById('out').textContent = 'WS parse error: ' + e;
    }
  };

  ws.onclose = () => setStatus('Disconnected', false);
  ws.onerror = () => setStatus('WebSocket error', true);
}

function sendHuman() {
  const sid = document.getElementById('sid').value.trim();
  const msg = document.getElementById('msg').value.trim();
  if (!sid || !msg) return alert('Session ID and message are required');

  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ message: msg }));
    document.getElementById('msg').value = '';
  } else {
    alert('Not connected. Click Connect first.');
  }
}

async function loadStatus() {
  const sid = document.getElementById('sid').value.trim();
  if (!sid) return alert('Enter Session ID first.');
  const res = await fetch('/chat/status?session_id=' + encodeURIComponent(sid));
  document.getElementById('out').textContent = await res.text();
}
</script>
</body>
</html>
"""
