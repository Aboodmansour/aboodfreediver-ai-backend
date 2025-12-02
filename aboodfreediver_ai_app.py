from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai.types import GenerateContentConfig
import os

# -----------------------------
# Gemini client configuration
# -----------------------------
# If you set GEMINI_API_KEY or GOOGLE_API_KEY in env, this is enough:
client = genai.Client()
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


def pick_urls_for_question(q: str) -> list[str]:
    """Very simple router: choose which URLs to give based on keywords."""
    q_low = q.lower()
    urls = set()

    # Always give homepage as a fallback
    home = "https://aboodfreediver.com"
    urls.add(home)

    # Prices
    if any(k in q_low for k in ["price", "cost", "fee", "discount"]):
        urls.add("https://aboodfreediver.com/prices.php")

    # Calendar / dates / availability
    if any(k in q_low for k in ["calendar", "schedule", "when", "date", "time",
                                "available", "availability", "booking", "book"]):
        urls.add("https://aboodfreediver.com/calendar.php")

    # Courses & training
    if any(k in q_low for k in ["course", "basic", "advanced", "discover",
                                "fun dive", "fun freediving", "training",
                                "session", "padi", "snorkel guide"]):
        urls.update({
            "https://aboodfreediver.com/courses.html",
            "https://aboodfreediver.com/Discoverfreediving.html",
            "https://aboodfreediver.com/BasicFreediver.html",
            "https://aboodfreediver.com/Freediver.html",
            "https://aboodfreediver.com/Advancedfreediver.html",
            "https://aboodfreediver.com/trainingsession.html",
            "https://aboodfreediver.com/FunFreediving.html",
            "https://aboodfreediver.com/snorkelguide.html",
        })

    # FAQ / requirements
    if any(k in q_low for k in ["faq", "question", "requirement", "requirements",
                                "age", "medical", "experience"]):
        urls.add("https://aboodfreediver.com/faq.html")

    # Contact
    if any(k in q_low for k in ["contact", "phone", "email", "whatsapp",
                                "reach you", "get in touch"]):
        urls.add("https://aboodfreediver.com/form1.php")

    # Equipment
    if any(k in q_low for k in ["equipment", "gear", "fins", "mask", "wetsuit"]):
        urls.add("https://aboodfreediver.com/freedivingequipment.html")

    # Photography
    if any(k in q_low for k in ["photo", "photography", "pictures", "photoshoot",
                                "camera", "underwater photos"]):
        urls.add("https://aboodfreediver.com/photographysession.html")

    # Dive sites / wrecks
    if any(k in q_low for k in ["dive site", "divesites", "wreck", "cedar pride",
                                "tank", "c-130", "tristar", "military"]):
        urls.update({
            "https://aboodfreediver.com/divesites.html",
            "https://aboodfreediver.com/cedar-pride.html",
            "https://aboodfreediver.com/military.html",
            "https://aboodfreediver.com/Tristar.html",
            "https://aboodfreediver.com/C-130.html",
            "https://aboodfreediver.com/Tank.html",
        })

    # Blog / tips / articles
    if any(k in q_low for k in ["blog", "article", "tip", "tips", "story", "stories"]):
        urls.update({
            "https://aboodfreediver.com/blog.html",
            "https://aboodfreediver.com/blog2.html",
            "https://aboodfreediver.com/blog3.html",
            "https://aboodfreediver.com/blog4.html",
        })

    # If we didn't detect anything special, just return a broad selection
    if len(urls) == 1:  # only homepage so far
        urls.update(BASE_URLS)

    urls_list = list(urls)

    # URL Context supports up to 20 URLs per request; enforce that limit.
    # Keep homepage first, then other URLs in any order up to 19 more.
    if len(urls_list) > 20:
        others = [u for u in urls_list if u != home]
        urls_list = [home] + others[:19]

    return urls_list


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
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=GenerateContentConfig(
                tools=[{"url_context": {}}]
            ),
        )

        answer = (response.text or "").strip() if response else ""

        if not answer:
            answer = (
                "Sorry, I could not read the website or generate an answer right now. "
                "Please try again or contact Abood Freediver directly."
            )

        return ChatResponse(answer=answer, source=MODEL_NAME)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")


@app.get("/")
async def root():
    return {"status": "ok", "service": "AI Mermaid (Gemini + URL Context)"}
