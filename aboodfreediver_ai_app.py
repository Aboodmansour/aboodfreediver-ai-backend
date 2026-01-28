from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai.types import GenerateContentConfig
import os
import time
import random
import logging
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# ============================================================
# CONFIG
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("freediver_assistant")

ASSISTANT_NAME = "Nerida"
ASSISTANT_ROLE = "Freediver Assistant"

MODEL_NAME = "gemini-2.5-flash"
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("Missing Gemini API key")

client = genai.Client(api_key=GEMINI_KEY)

# SMTP (IONOS)
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM")
OWNER_NOTIFY_EMAIL = os.getenv("OWNER_NOTIFY_EMAIL")

OWNER_ADMIN_TOKEN = os.getenv("OWNER_ADMIN_TOKEN", "")

MAX_URLS_PER_REQUEST = 8

# ============================================================
# EMAIL
# ============================================================

def send_email(subject: str, body: str):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, OWNER_NOTIFY_EMAIL]):
        logger.warning("SMTP not fully configured")
        return

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

RULE – HUMAN TAKEOVER:
If you are not confident, or the question needs exact confirmation,
start your reply with:

NEEDS_HUMAN: true

Then explain briefly why.
"""

def needs_human(text: str) -> bool:
    return "needs_human: true" in text.lower()

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

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")

    sid = get_session(req.session_id)
    add_msg(sid, "user", req.question)

    prompt = f"""
{SYSTEM_PROMPT}

Conversation:
{CONVERSATIONS[sid]["messages"]}

User question:
{req.question}
"""

    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=GenerateContentConfig()
        )
        answer = (resp.text or "").strip()

    except Exception as e:
        send_email(
            "[Nerida] AI ERROR",
            f"Session: {sid}\nError: {e}"
        )
        raise HTTPException(503, "AI unavailable")

    if not answer:
        answer = "NEEDS_HUMAN: true\nI could not answer this."

    add_msg(sid, "assistant", answer)
    flag = needs_human(answer)

    if flag:
        body = (
            f"Session: {sid}\n\n"
            f"FULL CONVERSATION:\n"
            f"{CONVERSATIONS[sid]['messages']}"
        )
        send_email("[Nerida] Human support needed", body)
        CONVERSATIONS[sid]["needs_human"] = True

    return ChatResponse(
        session_id=sid,
        answer=answer,
        needs_human=flag
    )
    
@app.get("/admin/conversations", dependencies=[Depends(admin_auth)])
async def admin_conversations():
    return CONVERSATIONS

@app.post("/human", dependencies=[Depends(admin_auth)])
async def human_reply(session_id: str, message: str):
    add_msg(session_id, "human", message)
    CONVERSATIONS[session_id]["needs_human"] = False
    return {"ok": True}
