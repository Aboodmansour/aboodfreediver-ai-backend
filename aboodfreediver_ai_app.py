from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Abood Freediver AI Mermaid", version="1.0.0")

# Allow calls from your website + local dev; keep "*" if you don't want to manage origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
origins = ["*"] if ALLOWED_ORIGINS.strip() == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models
# -----------------------------
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None  # [{role:"user"|"assistant", content:"..."}]
    email: Optional[str] = None  # optional follow-up email (front-end can send it if you add a field)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    source: str = "fallback"


# -----------------------------
# Optional: OpenAI (if OPENAI_API_KEY is set)
# -----------------------------
def _try_openai_chat(question: str, history: Optional[List[Dict[str, str]]]) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        return None

    # Lazy import so the app still runs if openai isn't installed.
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    client = OpenAI(api_key=api_key)

    system = (
        "You are AI Mermaid, the assistant for Abood Freediver (freediving courses in Aqaba, Jordan / Red Sea). "
        "Answer questions briefly and clearly. If the user asks about bookings, prices, or availability, "
        "direct them to the relevant site pages (Contact, Calendar, Prices)."
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]

    if history:
        # Keep only safe roles and limit size
        for m in history[-12:]:
            role = (m.get("role") or "").strip()
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=msgs,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.4")),
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None


# -----------------------------
# Fallback responder (so UI never hangs)
# -----------------------------
def _fallback_answer(question: str) -> str:
    q = question.strip().lower()

    if any(k in q for k in ["price", "prices", "cost", "how much", "fee"]):
        return "You can find current course prices here: https://www.aboodfreediver.com/Prices.php?lang=en"
    if any(k in q for k in ["calendar", "availability", "schedule", "date", "dates"]):
        return "Check upcoming trips and course dates here: https://www.aboodfreediver.com/calender.php"
    if any(k in q for k in ["contact", "email", "whatsapp", "phone", "book", "booking", "reserve"]):
        return "To book or ask details, please use the contact page: https://www.aboodfreediver.com/form1.php"
    if any(k in q for k in ["hello", "hi", "hey", "good morning", "good evening"]):
        return "Hi! Ask me about courses, prices, the calendar, safety, or what you can see while freediving in Aqaba."
    # generic
    return (
        "I can help with course info, prices, dates, safety, and recommendations for Aqaba/Red Sea. "
        "Ask a specific question (e.g., “How long is the Freediver course?”)."
    )


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    session_id = (req.session_id or "").strip() or str(uuid.uuid4())

    # Try OpenAI if configured, otherwise fallback.
    answer = _try_openai_chat(question, req.history)
    if answer:
        return ChatResponse(answer=answer, session_id=session_id, source="openai")

    return ChatResponse(answer=_fallback_answer(question), session_id=session_id, source="fallback")
