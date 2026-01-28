from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig
import os
import time
import random
import logging

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("ai_mermaid")
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Gemini client configuration
# -----------------------------
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# -----------------------------
# Known important URLs
# -----------------------------
BASE_URLS = [
    "https://aboodfreediver.com",                     # home
    "https://aboodfreediver.com/courses.html",        # courses overview
    "https://aboodfreediver.com/prices.php",          # prices page
    "https://aboodfreediver.com/calendar.php",        # calendar / schedule
    "https://aboodfreediver.com/faq.html",            # FAQ
    "https://aboodfreediver.com/form1.php",           # contact page

    # Course pages
    "https://aboodfreediver.com/Discoverfreediving.html",
    "https://aboodfreediver.com/BasicFreediver.html",
    "https://aboodfreediver.com/Freediver.html",
    "https://aboodfreediver.com/Advancedfreediver.html",
    "https://aboodfreediver.com/trainingsession.html",
    "https://aboodfreediver.com/FunFreediving.html",
    "https://aboodfreediver.com/snorkelguide.html",

    # Services / equipment / photography
    "https://aboodfreediver.com/freedivingequipment.html",
    "https://aboodfreediver.com/photographysession.html",

    # Dive sites
    "https://aboodfreediver.com/divesites.html",
    "https://aboodfreediver.com/cedar-pride.html",
    "https://aboodfreediver.com/military.html",
    "https://aboodfreediver.com/Tristar.html",
    "https://aboodfreediver.com/C-130.html",
    "https://aboodfreediver.com/Tank.html",

    # Blog articles
    "https://aboodfreediver.com/blog.html",
    "https://aboodfreediver.com/blog2.html",
    "https://aboodfreediver.com/blog3.html",
    "https://aboodfreediver.com/blog4.html",
]

# Limit URL Context payload size to reduce intermittent tool failures
MAX_URLS_PER_REQUEST = 8

def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def pick_urls_for_question(q: str) -> list[str]:
    """Deterministic router: choose which URLs to give based on keywords."""
    q_low = q.lower()

    home = "https://aboodfreediver.com"
    urls_list: list[str] = [home]  # keep deterministic order

    # Prices
    if any(k in q_low for k in ["price", "cost", "fee", "discount"]):
        urls_list.append("https://aboodfreediver.com/prices.php")

    # Calendar / dates / availability
    if any(k in q_low for k in ["calendar", "schedule", "when", "date", "time",
                                "available", "availability", "booking", "book"]):
        urls_list.append("https://aboodfreediver.com/calendar.php")

    # Courses & training
    if any(k in q_low for k in ["course", "basic", "advanced", "discover",
                                "fun dive", "fun freediving", "training",
                                "session", "padi", "snorkel guide"]):
        urls_list.extend([
            "https://aboodfreediver.com/courses.html",
            "https://aboodfreediver.com/Discoverfreediving.html",
            "https://aboodfreediver.com/BasicFreediver.html",
            "https://aboodfreediver.com/Freediver.html",
            "https://aboodfreediver.com/Advancedfreediver.html",
            "https://aboodfreediver.com/trainingsession.html",
            "https://aboodfreediver.com/FunFreediving.html",
            "https://aboodfreediver.com/snorkelguide.html",
        ])

    # FAQ / requirements
    if any(k in q_low for k in ["faq", "question", "requirement", "requirements",
                                "age", "medical", "experience"]):
        urls_list.append("https://aboodfreediver.com/faq.html")

    # Contact
    if any(k in q_low for k in ["contact", "phone", "email", "whatsapp",
                                "reach you", "get in touch"]):
        urls_list.append("https://aboodfreediver.com/form1.php")

    # Equipment
    if any(k in q_low for k in ["equipment", "gear", "fins", "mask", "wetsuit"]):
        urls_list.append("https://aboodfreediver.com/freedivingequipment.html")

    # Photography
    if any(k in q_low for k in ["photo", "photography", "pictures", "photoshoot",
                                "camera", "underwater photos"]):
        urls_list.append("https://aboodfreediver.com/photographysession.html")

    # Dive sites / wrecks
    if any(k in q_low for k in ["dive site", "divesites", "wreck", "cedar pride",
                                "tank", "c-130", "tristar", "military"]):
        urls_list.extend([
            "https://aboodfreediver.com/divesites.html",
            "https://aboodfreediver.com/cedar-pride.html",
            "https://aboodfreediver.com/military.html",
            "https://aboodfreediver.com/Tristar.html",
            "https://aboodfreediver.com/C-130.html",
            "https://aboodfreediver.com/Tank.html",
        ])

    # Blog / tips / articles
    if any(k in q_low for k in ["blog", "article", "tip", "tips", "story", "stories"]):
        urls_list.extend([
            "https://aboodfreediver.com/blog.html",
            "https://aboodfreediver.com/blog2.html",
            "https://aboodfreediver.com/blog3.html",
            "https://aboodfreediver.com/blog4.html",
        ])

    # If we didn't detect anything special, return a deterministic broad selection
    if len(urls_list) == 1:  # only homepage so far
        urls_list.extend(BASE_URLS)

    # Dedupe while preserving order
    urls_list = dedupe_preserve_order(urls_list)

    # Enforce URL Context limit (use a smaller cap for stability)
    # Always keep homepage first
    if urls_list and urls_list[0] != home:
        urls_list = [home] + [u for u in urls_list if u != home]

    return urls_list[:MAX_URLS_PER_REQUEST]

# -----------------------------
# Simple retry wrapper
# -----------------------------
def call_gemini_with_retry(callable_fn, attempts: int = 3, base_delay: float = 0.6):
    last_exc = None
    for i in range(attempts):
        try:
            return callable_fn()
        except Exception as e:
            last_exc = e
            sleep_s = base_delay * (2 ** i) + random.uniform(0, 0.25)
            time.sleep(sleep_s)
    raise last_exc

# -----------------------------
# System prompt
# -----------------------------
SYSTEM_PROMPT = """
You are "AI Mermaid", an AI employee and digital assistant for Abood Freediver
and the website aboodfreediver.com.

Identity and how to present yourself:
- When a user asks who you are, say that you are an AI employee / AI assistant
  for Abood Freediver, created to help visitors and students.
- When a user asks if you work for Abood Freediver, answer that you are an AI
  assistant working with Abood Freediver, not a human, but part of the Abood
  Freediver team to support them online.

Tasks:
- Answer questions about freediving technique, training, equalization, risks and safety.
- Answer questions about Abood Freediver services, courses, prices, calendar, and contact info.

Use the provided URLs with the URL Context tool to:
- Read the latest prices, course descriptions, calendar dates, and FAQ answers.
- Avoid hallucinating specific numbers or dates: always prefer what you actually read from the URLs.
- If something is not clearly on the site, say you're not sure and suggest the user contact Abood Freediver.

Language:
- Answer in the same language the user uses (English, German, French, Spanish, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Russian, Arabic).
- Keep answers short, clear and practical unless the user asks for more detail.
"""

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="AboodFreediver AI Mermaid API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # optionally restrict to ["https://aboodfreediver.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    source: str = MODEL_NAME

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    user_question = payload.question.strip()
    urls = pick_urls_for_question(user_question)
    urls_block = "\n".join(urls)

    prompt = f"""
{SYSTEM_PROMPT}

You have access to the following URLs from aboodfreediver.com.
Use the URL Context tool to read them and ground your answer
in the latest information from the site.

URLs:
{urls_block}

User question:
{user_question}
"""

    try:
        def _do():
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[{"url_context": {}}]
                ),
            )

        response = call_gemini_with_retry(_do, attempts=3)

        answer = (response.text or "").strip() if response else ""

        if not answer:
            answer = (
                "Sorry, I could not read the website or generate an answer right now. "
                "Please try again or contact Abood Freediver directly."
            )

        return ChatResponse(answer=answer, source=MODEL_NAME)

    except Exception as e:
        # Log full exception server-side, return stable message to client
        logger.exception("Gemini call failed. urls=%s question=%r", urls, user_question)
        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable. Please try again in a moment."
        )

@app.get("/")
async def root():
    return {"status": "ok", "service": "AI Mermaid (Gemini + URL Context)"}
