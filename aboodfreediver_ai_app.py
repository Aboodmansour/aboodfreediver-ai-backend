from __future__ import annotations

import os
import re
import uuid
import time
import json
import html as html_lib
import xml.etree.ElementTree as ET
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
    source: str = "fallback"  # gemini | gemini+site | gemini+blog | gemini+web | rules | fallback


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


def needs_human_confirmation(question: str) -> bool:
    q = (question or "").lower()
    return any(
        k in q
        for k in [
            # booking
            "book",
            "booking",
            "reserve",
            "reservation",
            # availability / dates
            "availability",
            "available",
            "calendar",
            "date",
            "dates",
            "schedule",
            "time slot",
            "are you free",
            "free time",
            "tomorrow",
            "today",
        ]
    )


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
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=15,
        )
        # 202 is success for SendGrid
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
# Sitemap + content extraction (site first, then blog)
# -----------------------------
SITEMAP_URL = _env("SITEMAP_URL", default="https://www.aboodfreediver.com/sitemaps.XML")
_sitemap_cache: Dict[str, Any] = {"ts": 0.0, "urls": []}

_fetch_cache: Dict[str, Any] = {}  # url -> {"ts": float, "text": str}


def _strip_html(html: str) -> str:
    # remove script/style
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    # remove tags
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_url_text(url: str, timeout: int = 12, max_chars: int = 12000, cache_seconds: int = 3600) -> str:
    if not url:
        return ""
    now = time.time()
    cached = _fetch_cache.get(url)
    if cached and (now - float(cached.get("ts", 0.0)) < cache_seconds) and isinstance(cached.get("text"), str):
        return cached["text"]

    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "AquaBot/1.0 (+https://www.aboodfreediver.com)"})
        r.raise_for_status()
        txt = _strip_html(r.text)
        if max_chars and len(txt) > max_chars:
            txt = txt[:max_chars] + " ..."
        _fetch_cache[url] = {"ts": now, "text": txt}
        return txt
    except Exception:
        _fetch_cache[url] = {"ts": now, "text": ""}
        return ""


def fetch_sitemap_urls() -> List[str]:
    """
    Fetches sitemap URLs. Cached 6 hours.
    Supports: sitemapindex + urlset.
    """
    now = time.time()
    if now - _sitemap_cache["ts"] < 6 * 3600 and isinstance(_sitemap_cache.get("urls"), list):
        return _sitemap_cache["urls"]

    urls: List[str] = []
    try:
        r = requests.get(SITEMAP_URL, timeout=12, headers={"User-Agent": "AquaBot/1.0 (+https://www.aboodfreediver.com)"})
        r.raise_for_status()

        root = ET.fromstring(r.text)

        def _tag_name(t: str) -> str:
            return t.split("}")[-1] if "}" in t else t

        if _tag_name(root.tag) == "sitemapindex":
            # pull child sitemaps and merge
            for sm in root.findall(".//{*}sitemap/{*}loc"):
                loc = (sm.text or "").strip()
                if loc:
                    try:
                        rr = requests.get(loc, timeout=12, headers={"User-Agent": "AquaBot/1.0 (+https://www.aboodfreediver.com)"})
                        rr.raise_for_status()
                        rr_root = ET.fromstring(rr.text)
                        for u in rr_root.findall(".//{*}url/{*}loc"):
                            uloc = (u.text or "").strip()
                            if uloc:
                                urls.append(uloc)
                    except Exception:
                        continue
        else:
            # urlset
            for u in root.findall(".//{*}url/{*}loc"):
                loc = (u.text or "").strip()
                if loc:
                    urls.append(loc)

        # de-dup preserve order
        urls = list(dict.fromkeys(urls))
        _sitemap_cache["ts"] = now
        _sitemap_cache["urls"] = urls
        return urls
    except Exception:
        _sitemap_cache["ts"] = now
        _sitemap_cache["urls"] = []
        return []


def _keyword_set(q: str) -> List[str]:
    q = (q or "").lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    parts = [p for p in q.split() if len(p) >= 3]
    # small de-dup, keep order
    out: List[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return out[:10]


def _score_url(url: str, keywords: List[str]) -> int:
    u = (url or "").lower()
    score = 0
    for k in keywords:
        if k in u:
            score += 3
    if "/blog" in u:
        score += 1
    return score


def gather_site_context(question: str, max_pages: int = 4) -> Tuple[str, List[str]]:
    """
    1) Use sitemap to pick likely pages.
    2) Fetch their text and return a compact context blob.
    """
    urls = fetch_sitemap_urls()
    if not urls:
        return ("", [])

    keywords = _keyword_set(question)
    candidates = sorted(urls, key=lambda u: _score_url(u, keywords), reverse=True)

    picked: List[str] = []
    snippets: List[str] = []
    for u in candidates:
        if len(picked) >= max_pages:
            break
        # avoid huge media files
        if re.search(r"\.(jpg|jpeg|png|gif|webp|pdf|mp4|mov)(\?|$)", u, re.IGNORECASE):
            continue
        # prefer main site first (not blog)
        if "/blog" in u.lower():
            continue
        text = fetch_url_text(u)
        if text:
            picked.append(u)
            snippets.append(f"PAGE: {u}\nCONTENT: {text}")

    context = "\n\n".join(snippets).strip()
    return (context, picked)


def gather_blog_context(question: str, max_pages: int = 4) -> Tuple[str, List[str]]:
    """
    Use sitemap to pick /blog pages; if none exist, returns empty.
    """
    urls = fetch_sitemap_urls()
    if not urls:
        return ("", [])

    blog_urls = [u for u in urls if "/blog" in (u or "").lower()]
    if not blog_urls:
        return ("", [])

    keywords = _keyword_set(question)
    candidates = sorted(blog_urls, key=lambda u: _score_url(u, keywords), reverse=True)

    picked: List[str] = []
    snippets: List[str] = []
    for u in candidates:
        if len(picked) >= max_pages:
            break
        text = fetch_url_text(u)
        if text:
            picked.append(u)
            snippets.append(f"BLOG: {u}\nCONTENT: {text}")

    context = "\n\n".join(snippets).strip()
    return (context, picked)


# -----------------------------
# SearchApi (web fallback)
# -----------------------------
def searchapi_key() -> str:
    # user asked: "is that SERPAPI_KEY?" -> accept both names + SearchApi naming
    return _env("SEARCHAPI_KEY", "SEARCHAPI_API_KEY", "SERPAPI_KEY", default="")


def searchapi_google(query: str, num: int = 5) -> List[Dict[str, str]]:
    """
    SearchApi.io Google engine.
    Returns list of {title, link, snippet}.
    """
    key = searchapi_key()
    if not key:
        return []

    try:
        r = requests.get(
            "https://www.searchapi.io/api/v1/search",
            params={"engine": "google", "q": query, "num": str(num), "api_key": key},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        organic = data.get("organic_results") or data.get("organic_results", [])
        results: List[Dict[str, str]] = []
        if isinstance(organic, list):
            for item in organic[:num]:
                if not isinstance(item, dict):
                    continue
                results.append(
                    {
                        "title": str(item.get("title", "")).strip(),
                        "link": str(item.get("link", "")).strip(),
                        "snippet": str(item.get("snippet", "")).strip(),
                    }
                )
        return results
    except Exception as e:
        print(f"SearchApi FAILED: {type(e).__name__}: {e}")
        return []


def build_searchapi_context(question: str, mode: str) -> Tuple[str, List[str]]:
    """
    mode:
      - site: limit to aboodfreediver.com
      - blog: limit to aboodfreediver.com + blog keyword
      - web: open web
    """
    domain = "aboodfreediver.com"
    if mode == "site":
        q = f"site:{domain} {question}"
    elif mode == "blog":
        q = f"site:{domain} blog {question}"
    else:
        q = question

    results = searchapi_google(q, num=5)
    if not results:
        return ("", [])

    lines: List[str] = []
    links: List[str] = []
    for r in results:
        title = r.get("title", "")
        link = r.get("link", "")
        snip = r.get("snippet", "")
        if link:
            links.append(link)
        lines.append(f"RESULT: {title}\nURL: {link}\nSNIPPET: {snip}")

    return ("\n\n".join(lines).strip(), links)


# -----------------------------
# Gemini (supports either new or old SDK)
# -----------------------------
def try_gemini_answer(question: str, history: Optional[List[Dict[str, str]]], extra_context: str = "") -> Optional[str]:
    api_key = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if not api_key:
        return None

    model = _env("GEMINI_MODEL", default="gemini-2.5-flash")

    base_rules = (
        "You are Aqua, the freediving assistant for Abood Freediver in Aqaba, Jordan (Red Sea).\n"
        "Behavior rules:\n"
        "1) Be brief, clear, helpful.\n"
        "2) If asked about prices, include this link: https://www.aboodfreediver.com/Prices.php?lang=en\n"
        "3) If asked about contact/booking, include this link: https://www.aboodfreediver.com/form1.php\n"
        "4) If asked about availability/dates:\n"
        "   - If calendar has events, mention next dates.\n"
        "   - If calendar has no events, say we are usually free BUT must confirm with the instructor.\n"
        "5) If uncertain, ask 1 short follow-up question.\n"
        "6) If provided with SITE/BLOG/WEB context, prioritize it over general knowledge.\n"
    )

    dates = fetch_calendar_events()
    cal_context = (
        "Upcoming dates from the calendar: " + ", ".join(dates)
        if dates
        else "Calendar shows no upcoming dates (may mean mostly free; must confirm with instructor)."
    )

    system = base_rules + "\n\n" + cal_context
    if extra_context and extra_context.strip():
        system += "\n\n" + "CONTEXT (may contain relevant excerpts):\n" + extra_context.strip()

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]

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
        gm = genai_old.GenerativeModel(model_name=model, system_instruction=system)

        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in msgs if m["role"] != "system"])
        r = gm.generate_content(prompt)
        text = getattr(r, "text", None)
        if text and str(text).strip():
            return str(text).strip()
    except Exception:
        return None

    return None


# -----------------------------
# Fallback logic (only if Gemini fails entirely)
# -----------------------------
def fallback_answer(question: str) -> Tuple[str, bool]:
    q = question.strip().lower()

    if any(k in q for k in ["price", "prices", "cost", "how much", "fee"]):
        return ("Prices are here: https://www.aboodfreediver.com/Prices.php?lang=en", False)

    FORM_URL = "https://www.aboodfreediver.com/form1.php"

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

    if any(k in q for k in ["availability", "available", "calendar", "date", "dates", "schedule", "time slot", "time are you free"]):
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
        hours = _env("OPENING_HOURS", default=f"Opening hours: please use the contact form to confirm today: {FORM_URL}")
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
    return {"ok": True, "time": _now_iso()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    session_id = (req.session_id or "").strip() or str(uuid.uuid4())

    convo = CONVERSATIONS.setdefault(
        session_id,
        {"messages": [], "needs_human": False, "created": _now_iso()},
    )
    convo["messages"].append({"role": "user", "text": question, "ts": _now_iso()})

    q = question.lower().strip()
    FORM_URL = "https://www.aboodfreediver.com/form1.php"

    # RULE OVERRIDES (fast deterministic, no notify)
    if any(k in q for k in ["open", "opening", "hours", "working hours", "what time do you open", "what time are you open"]):
        hours = _env("OPENING_HOURS", default="Open daily 9:00-17:00 (Aqaba time).")
        answer = hours
        convo["messages"].append({"role": "assistant", "text": answer, "ts": _now_iso()})
        return ChatResponse(answer=answer, session_id=session_id, needs_human=False, source="rules")

    if any(k in q for k in ["whatsapp", "phone", "contact", "email", "number"]):
        phone = _env("CONTACT_PHONE", default="")
        email = _env("CONTACT_EMAIL", default="free@aboodfreediver.com")
        whatsapp = _env("CONTACT_WHATSAPP", default=phone)

        msg = "You can contact us here:\n"
        if whatsapp:
            msg += f"Phone/WhatsApp: {whatsapp}\n"
        msg += f"Email: {email}\n"
        msg += f"Form: {FORM_URL}"

        convo["messages"].append({"role": "assistant", "text": msg, "ts": _now_iso()})
        return ChatResponse(answer=msg, session_id=session_id, needs_human=False, source="rules")

    # -----------------------------
    # MAIN LOGIC (Gemini must answer everything)
    # 1) search in site via sitemap pages
    # 2) search in blog pages via sitemap (if exists)
    # 3) search web via SearchApi
    # Then Gemini answers using the best context found.
    # -----------------------------
    site_ctx, site_urls = gather_site_context(question)
    if site_ctx:
        answer = try_gemini_answer(question, req.history, extra_context=site_ctx)
        if answer:
            needs_human = needs_human_confirmation(question)
            if needs_human:
                convo["needs_human"] = True
                send_owner_email(
                    subject="Aqua needs confirmation (booking / availability)",
                    body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
                )
            convo["messages"].append({"role": "assistant", "text": answer, "ts": _now_iso()})
            return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source="gemini+site")

    blog_ctx, blog_urls = gather_blog_context(question)
    if blog_ctx:
        answer = try_gemini_answer(question, req.history, extra_context=blog_ctx)
        if answer:
            needs_human = needs_human_confirmation(question)
            if needs_human:
                convo["needs_human"] = True
                send_owner_email(
                    subject="Aqua needs confirmation (booking / availability)",
                    body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
                )
            convo["messages"].append({"role": "assistant", "text": answer, "ts": _now_iso()})
            return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source="gemini+blog")

    # Web fallback using SearchApi (preferred because Render allows HTTPS)
    # First try site-only web search (acts like "search my site first" even if sitemap empty)
    web_site_ctx, _links_site = build_searchapi_context(question, mode="site")
    if web_site_ctx:
        answer = try_gemini_answer(question, req.history, extra_context=web_site_ctx)
        if answer:
            needs_human = needs_human_confirmation(question)
            if needs_human:
                convo["needs_human"] = True
                send_owner_email(
                    subject="Aqua needs confirmation (booking / availability)",
                    body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
                )
            convo["messages"].append({"role": "assistant", "text": answer, "ts": _now_iso()})
            return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source="gemini+site")

    # Then try blog-oriented search
    web_blog_ctx, _links_blog = build_searchapi_context(question, mode="blog")
    if web_blog_ctx:
        answer = try_gemini_answer(question, req.history, extra_context=web_blog_ctx)
        if answer:
            needs_human = needs_human_confirmation(question)
            if needs_human:
                convo["needs_human"] = True
                send_owner_email(
                    subject="Aqua needs confirmation (booking / availability)",
                    body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
                )
            convo["messages"].append({"role": "assistant", "text": answer, "ts": _now_iso()})
            return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source="gemini+blog")

    # Finally open web
    web_ctx, _links_web = build_searchapi_context(question, mode="web")
    if web_ctx:
        answer = try_gemini_answer(question, req.history, extra_context=web_ctx)
        if answer:
            needs_human = needs_human_confirmation(question)
            if needs_human:
                convo["needs_human"] = True
                send_owner_email(
                    subject="Aqua needs confirmation (booking / availability)",
                    body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
                )
            convo["messages"].append({"role": "assistant", "text": answer, "ts": _now_iso()})
            return ChatResponse(answer=answer, session_id=session_id, needs_human=needs_human, source="gemini+web")

    # If Gemini fails entirely (no key / SDK / etc.)
    # fallback text + optional human notify
    answer2, needs_human2 = fallback_answer(question)
    if needs_human2:
        convo["needs_human"] = True
        send_owner_email(
            subject="Aqua needs confirmation (booking / availability)",
            body=f"Session: {session_id}\n\nUser asked:\n{question}\n\nReply from admin page:\n/admin",
        )
    convo["messages"].append({"role": "assistant", "text": answer2, "ts": _now_iso()})
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

    convo["messages"].append({"role": "human", "text": data.message, "ts": _now_iso()})
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
