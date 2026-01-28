# aboodfreediver_ai_app.py
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig
import os
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# ============================================================
# CONFIG
# ============================================================

ASSISTANT_NAME = "Nerida"
MODEL_NAME = "gemini-2.5-flash"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    # Fail early so deployment shows a clear error in logs
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")

client = genai.Client(api_key=GEMINI_API_KEY)

# SMTP (IONOS)
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", "").strip()
OWNER_NOTIFY_EMAIL = os.getenv("OWNER_NOTIFY_EMAIL", "").strip()

OWNER_ADMIN_TOKEN = os.getenv("OWNER_ADMIN_TOKEN", "").strip()

# ============================================================
# EMAIL
# ============================================================

def send_email(subject: str, body: str) -> None:
    """
    Sends an email if SMTP is configured. Raises on failure.
    """
    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD and SMTP_FROM and OWNER_NOTIFY_EMAIL):
        raise RuntimeError("SMTP not configured (missing SMTP_* or OWNER_NOTIFY_EMAIL env vars)")

    msg = MIMEMultipart()
    msg["From"] = SMTP_FROM
    msg["To"] = OWNER_NOTIFY_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)

# ============================================================
# CONVERSATIONS (in-memory)
# ============================================================

CONVERSATIONS: Dict[str, Dict[str, Any]] = {}

def now() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_session(session_id: Optional[str]) -> str:
    if session_id and session_id in CONVERSATIONS:
        return session_id
    sid = session_id or str(uuid.uuid4())
    CONVERSATIONS[sid] = {
        "created": now(),
        "updated": now(),
        "messages": [],  # list[{role,text,ts}]
        "needs_human": False,
    }
    return sid

def add_msg(sid: str, role: str, text: str) -> None:
    CONVERSATIONS[sid]["messages"].append({
        "role": role,
        "text": text,
        "ts": now()
    })
    CONVERSATIONS[sid]["updated"] = now()

def render_messages_for_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Render in a stable, readable format for the model (better than dumping raw python objects).
    """
    lines = []
    for m in messages:
        role = m.get("role", "unknown")
        text = (m.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines).strip()

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are {ASSISTANT_NAME}, a professional Freediver Assistant.

IMPORTANT RULES:
- You must NOT confirm availability, prices, or bookings.
- If something requires confirmation, explain politely and suggest contacting the team.
- If user asks to book/reserve/confirm dates, ask for contact details and inform them a human will follow up.
"""

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Nerida – Freediver Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODELS
# ============================================================

class HistoryItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    history: Optional[List[HistoryItem]] = None  # optional, frontend can send it

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    needs_human: bool

def admin_auth(x_owner_token: Optional[str] = Header(None)):
    if not OWNER_ADMIN_TOKEN or x_owner_token != OWNER_ADMIN_TOKEN:
        raise HTTPException(401, "Unauthorized")

# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {"ok": True}

# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(400, "Empty question")

    sid = get_session(req.session_id)
    user_question = req.question.strip()

    # add user message
    add_msg(sid, "user", user_question)

    # If frontend sends its own chat history, you can optionally merge it.
    # To avoid duplication, we won't overwrite our server history; we only use server history for the model.
    convo_text = render_messages_for_prompt(CONVERSATIONS[sid]["messages"])

    prompt = f"""{SYSTEM_PROMPT}

Conversation so far:
{convo_text}

User question:
{user_question}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=GenerateContentConfig()
        )
    except Exception as e:
        raise HTTPException(502, f"AI provider error: {repr(e)}")

    answer = (getattr(response, "text", "") or "").strip()
    if not answer:
        answer = "Sorry — I could not generate a response. Please try again."

    add_msg(sid, "assistant", answer)

    # ========================================================
    # 🔔 BOOKING KEYWORDS → ALWAYS NOTIFY OWNER
    # ========================================================

    booking_keywords = [
        "book", "booking", "reserve", "availability",
        "tomorrow", "today", "confirm", "confirmation", "date"
    ]
    needs_human = any(k in user_question.lower() for k in booking_keywords)

    if needs_human:
        CONVERSATIONS[sid]["needs_human"] = True

        email_body = (
            f"Session ID: {sid}\n"
            f"Needs human: True\n\n"
            f"FULL CONVERSATION:\n\n"
            f"{render_messages_for_prompt(CONVERSATIONS[sid]['messages'])}\n"
        )

        # Do NOT break chat if email fails
        try:
            send_email(
                subject="[Nerida] Booking-related question (Human attention required)",
                body=email_body
            )
        except Exception as e:
            # log only
            print("Email failed:", repr(e))

    return ChatResponse(
        session_id=sid,
        answer=answer,
        needs_human=needs_human
    )

# ============================================================
# ADMIN ENDPOINTS
# ============================================================

@app.get("/admin/conversations", dependencies=[Depends(admin_auth)])
async def admin_conversations():
    return CONVERSATIONS

@app.post("/human", dependencies=[Depends(admin_auth)])
async def human_reply(session_id: str, message: str):
    if session_id not in CONVERSATIONS:
        raise HTTPException(404, "Unknown session_id")
    add_msg(session_id, "human", message)
    CONVERSATIONS[session_id]["needs_human"] = False
    return {"ok": True}
