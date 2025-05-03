import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import openai

# ──────────────────────────────────────────────────────────────────────────────
# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Advisor AI",
    description="Универсальный AI-консультант и ассистент на основе GPT-4",
    version="1.0.0",
)

# ──────────────────────────────────────────────────────────────────────────────
# Serve all files under ./static (CSS, JS, images, index.html)
# and enable default HTML responses for unknown routes
app.mount(
    "/static",
    StaticFiles(directory="static", html=True),
    name="static",
)
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def serve_index():
    """
    Return the main chat interface (static/index.html).
    """
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        return JSONResponse({"error": "static/index.html not found"}, status_code=404)
    return FileResponse(index_path, media_type="text/html")


@app.post("/chat")
async def chat(request: Request):
    """
    Receive a JSON payload with {"message": "..."} and proxy it to OpenAI GPT-4,
    then return {"reply": "..."}.
    """
    payload = await request.json()
    user_message = payload.get("message", "").strip()

    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — дружелюбный и компетентный AI-консультант Advisor. "
                        "Отвечай чётко, информативно и по существу."
                    )
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        assistant_reply = completion.choices[0].message.content.strip()
        return JSONResponse({"reply": assistant_reply})
    except Exception as e:
        # Log the exception if you have logging configured
        return JSONResponse({"error": f"OpenAI API error: {str(e)}"}, status_code=500)
