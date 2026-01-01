from fastapi import FastAPI
from api.core import router as ingest_router

app = FastAPI(title="RAG Service")
app.include_router(ingest_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
