from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import psutil
import os

app = FastAPI(
    title="Kupony Analityczne AI",
    description="API do analizy kuponów",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Stats(BaseModel):
    total_requests: int
    active_sessions: int
    avg_response_time: float
    last_update: datetime

class Monitor(BaseModel):
    cpu_usage: float
    memory_usage: float
    memory_available: int
    uptime: float
    timestamp: datetime

class Settings(BaseModel):
    debug: bool
    host: str
    port: int
    cors_enabled: bool
    version: str

@app.get("/")
async def root():
    return {
        "message": "Kupony Analityczne AI API",
        "version": "1.0.0",
        "endpoints": ["/", "/health", "/stats", "/monitor", "/settings"]
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/stats", response_model=Stats)
async def get_stats():
    return {
        "total_requests": 42,
        "active_sessions": 3,
        "avg_response_time": 0.125,
        "last_update": datetime.now()
    }

@app.get("/monitor", response_model=Monitor)
async def get_monitor():
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    return {
        "cpu_usage": cpu,
        "memory_usage": memory.percent,
        "memory_available": memory.available,
        "uptime": os.popen("uptime -p").read().strip(),
        "timestamp": datetime.now()
    }

@app.get("/settings", response_model=Settings)
async def get_settings():
    return {
        "debug": True,
        "host": "0.0.0.0",
        "port": 8000,
        "cors_enabled": True,
        "version": "1.0.0"
    }
