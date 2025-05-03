import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

# ——————————————————————————————————————————————————————————————————————————————
# 1) Загрузите .env и ваш ключ
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
# ——————————————————————————————————————————————————————————————————————————————

app = FastAPI(
    title="Advisor AI (GPT-4)",
    description="Универсальный AI-консультант на базе GPT-4",
    version="1.0.0",
)

# 2) Статика и главная страница
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/", response_class=FileResponse)
async def serve_index():
    path = os.path.join("static", "index.html")
    if not os.path.isfile(path):
        return JSONResponse({"error": "static/index.html not found"}, status_code=404)
    return FileResponse(path, media_type="text/html")

# 3) Инициализируем нового клиента OpenAI v1+
#    Обратите внимание: НЕ используем openai.ChatCompletion
client = OpenAI(api_key=api_key)

@app.post("/chat")
async def chat_endpoint(request: Request):
    """
    Ожидает JSON: {"message": "..."}
    Возвращает JSON: {"reply": "..."}
    """
    data = await request.json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    try:
        # Новый интерфейс openai-python v1+: client.chat.completions.create()
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Ты — компетентный AI-консультант Advisor. Отвечай по существу."},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        reply = resp.choices[0].message.content.strip()
        return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"error": f"OpenAI API error: {e}"}, status_code=500)
