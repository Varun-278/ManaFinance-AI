from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.chat import router as chat_router
from backend.api.routes.compare import router as compare_router
from backend.api.routes.health import router as health_router
from backend.api.routes.stocks import router as stocks_router
from backend.api.routes.trend import router as trend_router

app = FastAPI(title="ManaFinance-AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(stocks_router)
app.include_router(trend_router)
app.include_router(compare_router)
