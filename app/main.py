from fastapi import FastAPI
from app.routers.infer import router as infer_router

app = FastAPI(title="tree-health-infer-service", version="0.1.0")

@app.get("/health", summary="Health")
def health():
    return {"status": "ok"}

app.include_router(infer_router)
