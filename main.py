import os
import json
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# Загрузка переменных окружения
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

# Создание FastAPI приложения
app = FastAPI(
    title="Advisor AGI API",
    description="Универсальный AI-консультант с структурированными ответами",
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статические файлы
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# OpenAI клиент
client = OpenAI(api_key=api_key)

# Rate limiting
request_counts = defaultdict(list)
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60

def check_rate_limit(client_ip: str):
    """Проверка rate limiting"""
    now = datetime.now()
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip]
        if now - req_time < timedelta(seconds=RATE_LIMIT_WINDOW)
    ]
    
    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    request_counts[client_ip].append(now)
    return True

def load_json_data(filename: str):
    """Загрузка JSON данных из файла"""
    try:
        with open(f"data/{filename}", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": f"File {filename} not found"}
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON in {filename}"}

# Инициализация базы данных
def init_database():
    conn = sqlite3.connect("advisor_feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT NOT NULL,
            advisor_response TEXT NOT NULL,
            rating INTEGER,
            comment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_database()

# Модели данных
class ChatMessage(BaseModel):
    message: str

class FeedbackData(BaseModel):
    user_message: str
    advisor_response: str
    rating: int
    comment: str = ""

# Системный промпт
ADVISOR_SYSTEM_PROMPT = """
Ты — компетентный AI-консультант Advisor. Твоя задача — давать структурированные, этичные и объяснимые советы.

ОБЯЗАТЕЛЬНО структурируй свой ответ в следующем JSON-формате:

{
  "advice": "Краткий, конкретный совет по вопросу пользователя",
  "reasoning_path": "Пошаговое объяснение того, как ты пришел к этому решению. Опиши логику рассуждений",
  "ethical_check": "Анализ этических аспектов твоего совета. Соответствует ли он моральным принципам?",
  "self_reflection": "Твоя самооценка: насколько ты уверен в совете, есть ли ограничения или альтернативы?"
}

Отвечай ТОЛЬКО в этом JSON-формате. Каждое поле должно быть содержательным и полезным.
"""

# Эндпоинты
@app.get("/", response_class=FileResponse)
async def serve_index():
    """Главная страница"""
    path = os.path.join("static", "index.html")
    if not os.path.isfile(path):
        return JSONResponse({"error": "static/index.html not found"}, status_code=404)
    return FileResponse(path, media_type="text/html")

@app.post("/chat")
async def chat_endpoint(request: Request):
    """Старый эндпоинт для совместимости"""
    data = await request.json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Ты — компетентный AI-консультант Advisor. Отвечай по существу."},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        reply = resp.choices[0].message.content.strip()
        return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"error": f"OpenAI API error: {e}"}, status_code=500)

@app.post("/ask")
async def ask_advisor(message: ChatMessage, request: Request):
    """Новый эндпоинт со структурированными ответами"""
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
    
    user_message = message.message.strip()
    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1200,
        )
        
        response_text = resp.choices[0].message.content.strip()
        
        try:
            structured_response = json.loads(response_text)
            return JSONResponse(structured_response)
        except json.JSONDecodeError:
            return JSONResponse({
                "advice": response_text,
                "reasoning_path": "Ответ не был структурирован должным образом",
                "ethical_check": "Требуется дополнительная проверка",
                "self_reflection": "Система не смогла предоставить структурированный анализ"
            })
            
    except Exception as e:
        return JSONResponse({"error": f"OpenAI API error: {e}"}, status_code=500)

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackData):
    """Сохранение обратной связи"""
    try:
        conn = sqlite3.connect("advisor_feedback.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (user_message, advisor_response, rating, comment)
            VALUES (?, ?, ?, ?)
        """, (feedback.user_message, feedback.advisor_response, feedback.rating, feedback.comment))
        conn.commit()
        conn.close()
        
        return JSONResponse({"status": "success", "message": "Спасибо за обратную связь!"})
    except Exception as e:
        return JSONResponse({"error": f"Database error: {e}"}, status_code=500)

@app.get("/stats")
async def get_stats():
    """Статистика обратной связи"""
    try:
        conn = sqlite3.connect("advisor_feedback.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL")
        avg_rating = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE rating >= 4")
        positive_feedback = cursor.fetchone()[0]
        
        conn.close()
        
        return JSONResponse({
            "total_feedback": total_feedback,
            "average_rating": round(avg_rating, 2),
            "positive_feedback": positive_feedback,
            "satisfaction_rate": round((positive_feedback / total_feedback * 100), 2) if total_feedback > 0 else 0
        })
    except Exception as e:
        return JSONResponse({"error": f"Database error: {e}"}, status_code=500)

@app.get("/roadmap")
async def get_roadmap():
    """Получение roadmap проекта"""
    return JSONResponse(load_json_data("roadmap.json"))

@app.get("/user-stories")
async def get_user_stories():
    """Получение пользовательских историй"""
    return JSONResponse(load_json_data("user_stories.json"))

@app.get("/architecture")
async def get_architecture():
    """Получение информации об архитектуре"""
    return JSONResponse(load_json_data("architecture.json"))

@app.get("/legal")
async def get_legal():
    """Получение юридической информации"""
    return JSONResponse(load_json_data("legal.json"))

@app.get("/health")
async def health_check():
    """Проверка состояния системы"""
    try:
        conn = sqlite3.connect("advisor_feedback.db")
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        
        data_files = ["roadmap.json", "user_stories.json", "architecture.json", "legal.json"]
        missing_files = []
        for file in data_files:
            if not os.path.exists(f"data/{file}"):
                missing_files.append(file)
        
        status = "healthy" if not missing_files else "degraded"
        
        return JSONResponse({
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "missing_files": missing_files,
            "api_version": "2.0.0"
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "api_version": "2.0.0"
        }, status_code=503)

@app.get("/api/info")
async def api_info():
    """Информация об API"""
    return JSONResponse({
        "name": "Advisor AGI API",
        "version": "2.0.0",
        "description": "API для AI-консультанта с структурированными ответами",
        "endpoints": {
            "POST /ask": "Получить структурированный совет от AI",
            "POST /chat": "Старый эндпоинт для совместимости",
            "POST /feedback": "Отправить обратную связь",
            "GET /stats": "Получить статистику обратной связи",
            "GET /roadmap": "Получить roadmap проекта",
            "GET /user-stories": "Получить пользовательские истории",
            "GET /architecture": "Получить информацию об архитектуре",
            "GET /legal": "Получить юридическую информацию",
            "GET /health": "Проверить состояние системы",
            "GET /api/info": "Информация об API"
        },
        "rate_limits": {
            "requests_per_minute": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


