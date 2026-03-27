from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.database import init_db
from app.api.ingest import router as ingest_router
from app.api.query import router as query_router

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Academic RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest_router, prefix="/ingest", tags=["ingestion"])
app.include_router(query_router, prefix="/query", tags=["query"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def chat_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}
