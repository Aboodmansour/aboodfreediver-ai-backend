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


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _strip_html_to_text(raw_html: str) -> str:
    raw_html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw_html)
    raw_html = re.sub(r"(?is)<[^>]+>", " ", raw_html)
    raw_html = html_lib.unescape(raw_html)
    raw_html = re.sub(r"\s+", " ", raw_html).strip()
    return raw_html


def _tokenize(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) >= 3]


# -----------------------------
# Site/Blog/Web settings
# -----------------------------
BASE_SITE = _env("BASE_SITE_URL", "SITE_BASE_URL", default="https://www.aboodfreediver.com/")
if not BASE_SITE.endswith("/"):
    BASE_SITE += "/"

# Blog index (your env points to blog.html)
BLOG_INDEX_URL = _env("BLOG_SITEMAP_URL", default=urljoin(BASE_SITE, "blog.html"))

# Optional sitemap.xml (recommended; you already have SITE_SITEMAP_URL)
SITEMAP_URL = _env("SITE_SITEMAP_URL", default=urljoin(BASE_SITE, "sitemaps.XML"))

USER_AGENT = _env("AQUA_USER_AGENT", default="AquaBot/1.3 (+https://www.aboodfreediver.com)")
SHOW_SOURCES = _env_bool("SHOW_SOURCES", default=False)  # set true only for debugging

# Page cache
_PAGE_CACHE: Dict[str, Dict[str, Any]] = {}  # url -> {"ts": float, "text": str}
_CACHE_TTL_SECONDS = 60 * 30  # 30 minutes

# Sitemap cache
_SITEMAP_CACHE: Dict[str, Any] = {"ts": 0.0, "urls": []}
_SITEMAP_TTL_SECONDS = 60 * 60 * 6  # 6 hours


def fetch_page_text(url: str, timeout: int = 12) -> str:
    now = time.time()
    cached = _PAGE_CACHE.get(url)
    if cached and (now - float(cached.get("ts", 0)) < _CACHE_TTL_SECONDS):
        return str(cached.get("text", ""))

    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
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


def get_blog_urls(max_posts: int = 50) -> List[str]:
    try:
        r = requests.get(BLOG_INDEX_URL, timeout=12, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        links = extract_links(r.text, BLOG_INDEX_URL)
        blog_links = [u for u in links if re.search(r"/blog\d+\.html$", u, re.I)]
        return blog_links[:max_posts]
    except Exception:
        # fallback guesses
        return [urljoin(BASE_SITE, f"blog{i}.html") for i in range(1, 21)]


def get_main_site_urls() -> List[str]:
    # Keep this list, but we will ALSO use the sitemap so we don't miss pages.
    return [
        urljoin(BASE_SITE, "index.html"),
        urljoin(BASE_SITE, "FAQ.html"),
        urljoin(BASE_SITE, "Discoverfreediving.html"),
        urljoin(BASE_SITE, "BasicFreediver.html"),
        urljoin(BASE_SITE, "Freediver.html"),
        urljoin(BASE_SITE, "Advancedfreediver.html"),
        urljoin(BASE_SITE, "trainingsession.html"),
        urljoin(BASE_SITE, "FunFreediving.html"),
        urljoin(BASE_SITE, "freedivingequipment.html"),
        urljoin(BASE_SITE, "divesites.html"),
        urljoin(BASE_SITE, "cedar-pride.html"),
        urljoin(BASE_SITE, "military.html"),
        urljoin(BASE_SITE, "Tristar.html"),
        urljoin(BASE_SITE, "C-130.html"),
        urljoin(BASE_SITE, "Tank.html"),
    ]


def _parse_sitemap_urls(xml_text: str) -> List[str]:
    # Simple sitemap parser without extra deps
    # Works for standard <urlset><url><loc>...</loc>...
    locs = re.findall(r"(?is)<loc>\s*([^<\s]+)\s*</loc>", xml_text)
    urls = []
    for u in locs:
        u = u.strip()
        if u.startswith(BASE_SITE):
            urls.append(u)
    return list(dict.fromkeys(urls))


def get_sitemap_urls(limit: int = 300) -> List[str]:
    now = time.time()
    if now - float(_SITEMAP_CACHE.get("ts", 0.0)) < _SITEMAP_TTL_SECONDS and isinstance(_SITEMAP_CACHE.get("urls"), list):
        urls = _SITEMAP_CACHE["urls"]
        return urls[:limit]

    urls: List[str] = []
    try:
        r = requests.get(SITEMAP_URL, timeout=15, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        urls = _parse_sitemap_urls(r.text)
    except Exception:
        urls = []

    # cache even if empty (avoid hammering)
    _SITEMAP_CACHE["ts"] = now
    _SITEMAP_CACHE["urls"] = urls
    return urls[:limit]


def _candidate_urls(question: str) -> List[str]:
    terms = _tokenize(question)
    # Always include these
    base = get_main_site_urls()
    blog = get_blog_urls(max_posts=30)

    # From sitemap, pick likely URLs first by URL string match
    sm = get_sitemap_urls(limit=500)
    ranked: List[Tuple[int, str]] = []
    for u in sm:
        score = 0
        lu = u.lower()
        for t in set(terms):
            if t in lu:
                score += 2
        # Small bias for likely content pages
        if lu.endswith(".html"):
            score += 1
        if score > 0:
            ranked.append((score, u))
    ranked.sort(key=lambda x: x[0], reverse=True)
    sm_candidates = [u for _, u in ranked[:60]]  # keep it bounded

    # De-dup, preserve order
    out: List[str] = []
    for u in base + sm_candidates + blog:
        if u.startswith(BASE_SITE) and u not in out:
            out.append(u)

    return out


def score_text_match(query: str, text: str) -> int:
    q_terms = _tokenize(query)
    if not q_terms:
        return 0
    t = (text or "").lower()
    score = 0
    for term in set(q_terms):
        if term in t:
            score += 1
    return score


def retrieve_site_context(
    question: str,
    urls: List[str],
    top_k: int = 6,
    max_chars: int = 6500,
) -> List[Dict[str, str]]:
    scored: List[Tuple[int, str, str]] = []
    for u in urls:
        txt = fetch_page_text(u)
        if not txt:
            continue
        s = score_text_match(question, txt)
        if s <= 0:
            continue
        scored.append((s, u, txt))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, str]] = []
    for s, u, txt in scored[:top_k]:
        snippet = txt[: min(len(txt), 1500)]
        out.append({"url": u, "snippet": snippet})

    # Trim combined size
    total = 0
    trimmed: List[Dict[str, str]] = []
    for item in out:
        chunk = f"URL: {item['url']}\nTEXT: {item['snippet']}\n\n"
        if total + len(chunk) > max_chars:
            break
        trimmed.append(item)
        total += len(chunk)
    return trimmed


def searchapi_web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """
    Uses SearchApi.io Google Search API.
    Docs: https://www.searchapi.io/docs/google
    """
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
        return out
    except Exception:
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
    now = time.time()
    if now - float(_calendar_cache.get("ts", 0.0)) < 600 and isinstance(_calendar_cache.get("events"), list):
        return _calendar_cache["events"]

    try:
        r = requests.get(CALENDAR_URL, timeout=10, headers={"User-Agent": USER_AGENT})
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
# Intent detection (minimal rules)
# -----------------------------
def needs_human_confirmation(question: str) -> bool:
    q = (question or "").lower()
    # Only use human-confirm flow for booking/availability-like requests
    return any(
        k in q
        for k in [
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
    )


# -----------------------------
# Gemini with grounding (site -> web fallback)
# -----------------------------
def try_gemini_answer(question: str, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        return None

    model = _env("GEMINI_MODEL", default="gemini-2.5-flash")

    # 1) Site grounding (uses sitemap candidates so you don't miss pages like Freediver.html prerequisites)
    candidates = _candidate_urls(question)
    site_ctx = retrieve_site_context(question, candidates, top_k=6)

    # 2) Web grounding only if site didn't match
    web_ctx: List[Dict[str, str]] = []
    if not site_ctx:
        web_ctx = searchapi_web_search(question, k=5)

    dates = fetch_calendar_events()
    cal_context = (
        "Upcoming dates: " + ", ".join(dates)
        if dates
        else "Calendar: no upcoming dates listed (must confirm with instructor)."
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

    # IMPORTANT: user wants direct answers (no “I found…” / no link-dumps)
    # If SHOW_SOURCES=1, we will append sources at the very end for debugging only.
    system = (
        "You are Aqua, the freediving assistant for Abood Freediver in Aqaba, Jordan.\n"
        "Answer the user's question directly and clearly in 1–6 sentences.\n"
        "Do NOT say phrases like: 'I found relevant info', 'I couldn’t find this', 'Try opening the page'.\n"
        "Do NOT list URLs unless the user explicitly asks for a link.\n"
        "If you used the provided sources, silently use them to answer.\n"
        "If the answer is not supported by sources, give a best-effort general answer without mentioning that sources are missing.\n\n"
        f"{cal_context}\n"
    )

    # Build prompt
    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system + ("\n\n" + grounding_text if grounding_text else "")}
    ]

    if history:
        for m in history[-10:]:
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

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 350},
        )
        text = getattr(resp, "text", None)
        if text and str(text).strip():
            out = str(text).strip()
            if SHOW_SOURCES and site_ctx:
                out += "\n\n(Sources used: " + ", ".join([c["url"] for c in site_ctx[:3]]) + ")"
            return out
    except Exception as e:
        # Make the 429 visible in logs, but do NOT break the user flow
        print(f"Gemini FAILED (new SDK): {type(e).__name__}: {e}")

    # --- Fallback old SDK: google-generativeai ---
    try:
        import google.generativeai as genai_old  # type: ignore

        genai_old.configure(api_key=api_key)
        gm = genai_old.GenerativeModel(
            model_name=model,
            system_instruction=system + ("\n\n" + grounding_text if grounding_text else ""),
        )

        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs if m["role"] != "system"])
        r = gm.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 350},
        )
        text = getattr(r, "text", None)
        if text and str(text).strip():
            out = str(text).strip()
            if SHOW_SOURCES and site_ctx:
                out += "\n\n(Sources used: " + ", ".join([c["url"] for c in site_ctx[:3]]) + ")"
            return out
    except Exception as e:
        print(f"Gemini FAILED (old SDK): {type(e).__name__}: {e}")

    return None


# -----------------------------
# Fallback answer (when Gemini quota/timeouts)
# -----------------------------
def fallback_answer(question: str) -> str:
    q = (question or "").strip().lower()

    # Minimal helpful fallbacks
    if any(k in q for k in ["price", "prices", "cost", "how much", "fee"]):
        return "You can see prices on the Prices page."

    if any(k in q for k in ["whatsapp", "phone", "contact", "email", "number"]):
        phone = _env("CONTACT_PHONE", default="")
        email = _env("CONTACT_EMAIL", default="free@aboodfreediver.com")
        whatsapp = _env("CONTACT_WHATSAPP", default=phone)
        parts = []
        if whatsapp:
            parts.append(f"Phone/WhatsApp: {whatsapp}")
        if email:
            parts.append(f"Email: {email}")
        return " | ".join(parts) if parts else "You can contact us via the website contact form."

    # Generic
    return "Tell me what you want to do (beginner try, course, fun dive, or training days) and I’ll recommend the best option."


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

    # Determine if this needs human confirmation
    needs_human = needs_human_confirmation(question)

    # Try Gemini (site-grounded; web fallback)
    answer = try_gemini_answer(question, req.history)

    if answer:
        if needs_human:
            convo["needs_human"] = True
            send_owner_email(
                subject="Aqua needs confirmation (booking/availability)",
                body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
            )
        convo["messages"].append({"role": "assistant", "text": answer, "ts": datetime.now(timezone.utc).isoformat()})
        return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source="gemini")

    # Gemini failed (quota/timeout/etc.) -> simple fallback answer
    answer2 = fallback_answer(question)
    if needs_human:
        convo["needs_human"] = True
        send_owner_email(
            subject="Aqua needs confirmation (booking/availability)",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
        )

    convo["messages"].append({"role": "assistant", "text": answer2, "ts": datetime.now(timezone.utc).isoformat()})
    return ChatResponse(answer=answer2, session_id=session_id, needs_human=needs_human, source="fallback")


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
