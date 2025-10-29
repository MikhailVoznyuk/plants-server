# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app import config
from app.routers.infer import router as infer_router

app = FastAPI(title="tree-health-infer-service", version="0.1.0")

# если фронт на другом домене/порту — не жадничай, включи CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # или список твоих доменов
    allow_methods=["*"],
    allow_headers=["*"],
)

# вот это главное: публикуем /out -> config.OUT_DIR (обычно /data/out)
app.mount("/out", StaticFiles(directory=config.OUT_DIR), name="out")

@app.get("/health", summary="Health")
def health():
    return {"status": "ok"}

app.include_router(infer_router)
