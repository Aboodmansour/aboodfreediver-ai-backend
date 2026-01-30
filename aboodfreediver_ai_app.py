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
app = FastAPI(title="Aqua – Abood Freediver Assistant", version="1.2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

DEBUG = os.getenv("DEBUG", "").strip() in ("1", "true", "True", "yes", "YES")


def _debug(*args: Any) -> None:
    if DEBUG:
        print("[DEBUG]", *args)


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


# -----------------------------
# Site/Blog Retrieval Helpers (sitemap -> site pages -> blog -> web)
# -----------------------------
BASE_SITE = _env("SITE_BASE_URL", "BASE_SITE_URL", default="https://www.aboodfreediver.com/")
if not BASE_SITE.endswith("/"):
    BASE_SITE += "/"

SITE_SITEMAP_URL = _env("SITE_SITEMAP_URL", default=urljoin(BASE_SITE, "sitemaps.XML"))
BLOG_INDEX_URL = _env("BLOG_SITEMAP_URL", default=urljoin(BASE_SITE, "blog.html"))

_PAGE_CACHE: Dict[str, Dict[str, Any]] = {}  # url -> {"ts": float, "text": str}
_CACHE_TTL_SECONDS = 60 * 60  # 1 hour

_SITEMAP_CACHE: Dict[str, Any] = {"ts": 0.0, "urls": []}
_SITEMAP_TTL_SECONDS = 12 * 60 * 60  # 12 hours

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def searchapi_web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """
    Uses SearchApi.io Google Search API.
    Env: SEARCHAPI_KEY
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
    except Exception as e:
        _debug("SearchApi failed:", type(e).__name__, str(e)[:200])
        return []


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
        r = requests.get(url, timeout=timeout, headers={"User-Agent": DEFAULT_UA})
        r.raise_for_status()
        text = _strip_html_to_text(r.text)
        _debug("Fetched page:", url, "chars:", len(text))
    except Exception as e:
        _debug("Fetch failed:", url, type(e).__name__, str(e)[:200])
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


def get_site_urls_from_sitemap(max_urls: int = 120) -> List[str]:
    now = time.time()
    if now - float(_SITEMAP_CACHE.get("ts", 0)) < _SITEMAP_TTL_SECONDS and isinstance(_SITEMAP_CACHE.get("urls"), list):
        return list(_SITEMAP_CACHE["urls"])[:max_urls]

    urls: List[str] = []
    try:
        r = requests.get(SITE_SITEMAP_URL, timeout=15, headers={"User-Agent": DEFAULT_UA})
        r.raise_for_status()
        xml = r.text or ""
        locs = re.findall(r"(?is)<loc>\s*([^<\s]+)\s*</loc>", xml)
        for u in locs:
            u = u.strip()
            if u.startswith(BASE_SITE):
                urls.append(u)
        urls = list(dict.fromkeys(urls))[:max_urls]
        _debug("Sitemap URLs:", len(urls))
    except Exception as e:
        _debug("Sitemap fetch failed:", type(e).__name__, str(e)[:200])
        urls = []

    _SITEMAP_CACHE["ts"] = now
    _SITEMAP_CACHE["urls"] = urls
    return urls


def get_blog_urls(max_posts: int = 30) -> List[str]:
    try:
        r = requests.get(BLOG_INDEX_URL, timeout=15, headers={"User-Agent": DEFAULT_UA})
        r.raise_for_status()
        links = extract_links(r.text, BLOG_INDEX_URL)
        blog_links = [u for u in links if re.search(r"/blog\d+\.html$", u, re.I)]
        return blog_links[:max_posts]
    except Exception as e:
        _debug("Blog index fetch failed:", type(e).__name__, str(e)[:200])
        return [urljoin(BASE_SITE, f"blog{i}.html") for i in range(1, 21)]


def _normalize_tokens(s: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) >= 3]


def score_text_match(query: str, text: str) -> int:
    """
    Improved scoring:
    - Adds synonyms so "age required" can match "18 years" + "pre-requisites"
    - Gives extra weight to key terms
    """
    if not text:
        return 0

    tokens = _normalize_tokens(query)
    # add helpful synonyms
    if any(t in tokens for t in ["age", "years", "year"]):
        tokens += ["years", "year", "minimum", "min", "old"]
    if any(t in tokens for t in ["required", "require", "requirements", "prerequisite", "prerequisites"]):
        tokens += ["prerequisite", "prerequisites", "requirements", "pre", "requisites", "requisite", "must"]

    t = text.lower()
    score = 0
    for term in set(tokens):
        if term in t:
            score += 1

    # small boost if course name is in the page
    if "freediver" in (query or "").lower() and "freediver" in t:
        score += 2

    return score


def retrieve_site_context(question: str, urls: List[str], top_k: int = 5, max_chars: int = 4500) -> List[Dict[str, str]]:
    scored: List[Tuple[int, str, str]] = []
    for u in urls:
        txt = fetch_page_text(u)
        s = score_text_match(question, txt)
        if s <= 0:
            continue
        scored.append((s, u, txt))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, str]] = []
    for s, u, txt in scored[:top_k]:
        out.append({"url": u, "snippet": txt[: min(len(txt), 1400)]})

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
        return _calendar_cache["events"]

    try:
        r = requests.get(CALENDAR_URL, timeout=12, headers={"User-Agent": DEFAULT_UA})
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
# Gemini (sitemap -> blog -> web)
# -----------------------------
def try_gemini_answer(question: str, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        return None

    model = _env("GEMINI_MODEL", default="gemini-2.5-flash")

    # 1) Site pages from sitemap (best coverage)
    site_urls = get_site_urls_from_sitemap(max_urls=120)
    main_ctx = retrieve_site_context(question, site_urls)

    # 2) Blog if site didn't match
    blog_ctx: List[Dict[str, str]] = []
    if not main_ctx:
        blog_ctx = retrieve_site_context(question, get_blog_urls())

    have_grounding = bool(main_ctx or blog_ctx)

    # 3) Web only if nothing matched on site/blog
    web_ctx: List[Dict[str, str]] = []
    if not have_grounding:
        web_ctx = searchapi_web_search(question, k=5)

    _debug("Grounding:", "site", len(main_ctx), "blog", len(blog_ctx), "web", len(web_ctx))

    dates = fetch_calendar_events()
    cal_context = (
        "Upcoming dates from the calendar: " + ", ".join(dates)
        if dates
        else "Calendar shows no upcoming dates (may mean mostly free; must confirm with instructor)."
    )

    grounding_text = ""
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
        "You are Aqua, the freediving assistant for Abood Freediver in Aqaba, Jordan (Red Sea).\n"
        "Always prioritize safety. If the user asks for medical advice, recommend seeing a professional.\n\n"
        "Hard business rules:\n"
        "1) If asked about prices, link to: https://www.aboodfreediver.com/Prices.php?lang=en\n"
        "2) If asked about contact/booking, link to: https://www.aboodfreediver.com/form1.php\n"
        "3) If asked about availability/dates:\n"
        "   - If calendar has events, mention the next dates.\n"
        "   - If calendar has no events, say we are usually free BUT must confirm with the instructor.\n\n"
    )

    if have_grounding or web_ctx:
        system += (
            "Content rules (IMPORTANT):\n"
            "- Use ONLY the provided MAIN SITE / BLOG SOURCES and (if present) WEB RESULTS for factual claims.\n"
            "- If the answer is not in those sources, say exactly: \"I couldn’t find this on the website/blog.\" "
            "Then provide a short best-effort general answer.\n"
            "- When you use info from sources, cite the URL(s) in the answer.\n"
        )
    else:
        system += "Provide a concise, accurate general answer.\n"

    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system + "\n\n" + cal_context + ("\n\n" + grounding_text if grounding_text else "")}
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

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 700},
        )
        text = getattr(resp, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception as e:
        _debug("New Gemini SDK failed:", type(e).__name__, str(e)[:200])

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
        _debug("Old Gemini SDK failed:", type(e).__name__, str(e)[:200])
        return None

    return None


# -----------------------------
# Fallback logic
# -----------------------------
def fallback_answer(question: str) -> Tuple[str, bool]:
    q = question.strip().lower()

    if any(k in q for k in ["price", "prices", "cost", "how much", "fee"]):
        return ("Prices are here: https://www.aboodfreediver.com/Prices.php?lang=en", False)

    form_url = "https://www.aboodfreediver.com/form1.php"

    if any(k in q for k in ["book", "booking", "reserve", "reservation"]):
        return (
            f"To book, please fill this form: {form_url} "
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
        msg += f"Form: {form_url}"
        return (msg, False)

    if any(k in q for k in ["availability", "available", "calendar", "date", "dates", "schedule", "time slot", "time are you free"]):
        dates = fetch_calendar_events()
        if dates:
            return ("Upcoming calendar dates: " + ", ".join(dates) + ". Tell me the day/time you want and I’ll confirm.", True)
        return ("We’re usually free if nothing is scheduled, but I must confirm first. Tell me the day/time you want.", True)

    if any(k in q for k in ["open", "opening", "hours", "working hours", "what time do you open", "what time are you open"]):
        hours = _env("OPENING_HOURS", default=f"Opening hours: please use the contact form to confirm today: {form_url}")
        return (hours, False)

    if any(k in q for k in ["hello", "hi", "hey", "good morning", "good evening"]):
        return ("Hi, I’m Aqua. Ask me about courses, prices, safety, or availability in Aqaba.", False)

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

    convo = CONVERSATIONS.setdefault(
        session_id,
        {"messages": [], "needs_human": False, "created": datetime.now(timezone.utc).isoformat()},
    )
    convo["messages"].append({"role": "user", "text": question, "ts": datetime.now(timezone.utc).isoformat()})

    q = question.lower().strip()
    form_url = "https://www.aboodfreediver.com/form1.php"

    # RULE OVERRIDES (NO NOTIFY)
    if any(k in q for k in ["open", "opening", "hours", "working hours", "what time do you open", "what time are you open"]):
        hours = _env("OPENING_HOURS", default="Open daily 9:00-17:00 (Aqaba time).")
        return ChatResponse(answer=hours, session_id=session_id, needs_human=False, source="rules")

    if any(k in q for k in ["whatsapp", "phone", "contact", "email", "number"]):
        phone = _env("CONTACT_PHONE", default="")
        email = _env("CONTACT_EMAIL", default="free@aboodfreediver.com")
        whatsapp = _env("CONTACT_WHATSAPP", default=phone)

        msg = "You can contact us here:\n"
        if whatsapp:
            msg += f"Phone/WhatsApp: {whatsapp}\n"
        msg += f"Email: {email}\n"
        msg += f"Form: {form_url}"
        return ChatResponse(answer=msg, session_id=session_id, needs_human=False, source="rules")

    # Courses: ONLY show the generic list if the user is not asking about requirements
    if any(k in q for k in ["courses", "course", "levels", "learn", "training", "certification"]):
        requirement_terms = [
            "age", "old", "minimum", "min", "years",
            "require", "required", "requirements", "prerequisite", "prerequisites",
            "medical", "doctor", "fit", "fitness", "health",
        ]
        if not any(t in q for t in requirement_terms):
            msg = (
                "We offer freediving courses for all levels:\n"
                "- Discovery Freediver (beginner try)\n"
                "- Freediver (Level 1)\n"
                "- Advanced Freediver (Level 2)\n"
                "- Master Freediver (Level 3)\n\n"
                "Tell me your experience level and how many days you have, and I’ll recommend the best option."
            )
            return ChatResponse(answer=msg, session_id=session_id, needs_human=False, source="rules")
        # else: fall through to Gemini so it can fetch prerequisites from the site

    # try gemini
    answer = try_gemini_answer(question, req.history)
    if answer:
        needs_human = any(
            k in q
            for k in ["availability", "available", "calendar", "date", "dates", "schedule", "time slot", "book", "booking", "reserve", "reservation"]
        )
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
