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
from typing import Optional, Dict, Any

# ============================================================
# CONFIG
# ============================================================

ASSISTANT_NAME = "Nerida"
MODEL_NAME = "gemini-2.5-flash"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# SMTP (IONOS)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM")
OWNER_NOTIFY_EMAIL = os.getenv("OWNER_NOTIFY_EMAIL")

OWNER_ADMIN_TOKEN = os.getenv("OWNER_ADMIN_TOKEN", "")

# ============================================================
# EMAIL
# ============================================================

def send_email(subject: str, body: str):
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

def now():
    return datetime.now(timezone.utc).isoformat()

def get_session(session_id: Optional[str]) -> str:
    if session_id and session_id in CONVERSATIONS:
        return session_id
    sid = session_id or str(uuid.uuid4())
    CONVERSATIONS[sid] = {
        "created": now(),
        "updated": now(),
        "messages": [],
        "needs_human": False,
    }
    return sid

def add_msg(sid: str, role: str, text: str):
    CONVERSATIONS[sid]["messages"].append({
        "role": role,
        "text": text,
        "ts": now()
    })
    CONVERSATIONS[sid]["updated"] = now()

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are {ASSISTANT_NAME}, a professional Freediver Assistant.

IMPORTANT RULE:
- You must NOT confirm availability, prices, or bookings.
- If something requires confirmation, explain politely.
"""

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Nerida – Freediver Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    needs_human: bool

def admin_auth(x_owner_token: Optional[str] = Header(None)):
    if x_owner_token != OWNER_ADMIN_TOKEN:
        raise HTTPException(401, "Unauthorized")

# ============================================================
# CHAT ENDPOINT (UPDATED)
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")

    sid = get_session(req.session_id)
    user_question = req.question.strip()

    add_msg(sid, "user", user_question)

    prompt = f"""
{SYSTEM_PROMPT}

Conversation so far:
{CONVERSATIONS[sid]["messages"]}

User question:
{user_question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=GenerateContentConfig()
    )

    answer = (response.text or "").strip()
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
            f"Session ID: {sid}\n\n"
            f"FULL CONVERSATION:\n\n"
            f"{CONVERSATIONS[sid]['messages']}"
        )

        send_email(
            subject="[Nerida] Booking-related question (Human attention required)",
            body=email_body
        )

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
    add_msg(session_id, "human", message)
    CONVERSATIONS[session_id]["needs_human"] = False
    return {"ok": True}
