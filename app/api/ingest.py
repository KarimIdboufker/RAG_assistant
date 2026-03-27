from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, get_db
from app.ingestion.pipeline import ingest_pdf
from app.models import Chunk, Paper

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PaperOut(BaseModel):
    id: str
    filename: str
    title: str | None
    authors: str | None
    chunk_count: int

    class Config:
        from_attributes = True


class IngestStatus(BaseModel):
    message: str
    total_pdfs: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/all", response_model=IngestStatus)
def ingest_all(background_tasks: BackgroundTasks):
    """Trigger background ingestion of all PDFs in the configured pdf_dir."""
    pdf_dir = Path(settings.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(status_code=404, detail=f"No PDFs found in {pdf_dir}")

    background_tasks.add_task(_run_ingestion, [str(p) for p in pdfs])
    return IngestStatus(message="Ingestion started", total_pdfs=len(pdfs))


@router.post("/{filename}", response_model=PaperOut)
def ingest_one(filename: str, db: Session = Depends(get_db)):
    """Ingest a single PDF synchronously (useful for testing)."""
    pdf_path = Path(settings.pdf_dir) / filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found")

    paper, chunk_count = ingest_pdf(str(pdf_path), db)
    return _paper_out(paper, chunk_count)


@router.get("/papers", response_model=list[PaperOut])
def list_papers(db: Session = Depends(get_db)):
    """List all ingested papers with their chunk counts."""
    papers = db.query(Paper).order_by(Paper.ingested_at.desc()).all()
    return [
        _paper_out(p, db.query(Chunk).filter_by(paper_id=p.id).count())
        for p in papers
    ]


@router.delete("/papers/{paper_id}", status_code=204)
def delete_paper(paper_id: str, db: Session = Depends(get_db)):
    """Remove a paper and all its chunks."""
    paper = db.query(Paper).filter_by(id=paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    db.delete(paper)
    db.commit()


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _run_ingestion(pdf_paths: list[str]):
    db = SessionLocal()
    try:
        for path in pdf_paths:
            try:
                paper, n = ingest_pdf(path, db)
                print(f"[ingest] {Path(path).name} → {n} chunks  ({paper.title or 'no title'})")
            except Exception as exc:
                print(f"[ingest] ERROR {Path(path).name}: {exc}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper_out(paper: Paper, chunk_count: int) -> PaperOut:
    return PaperOut(
        id=str(paper.id),
        filename=paper.filename,
        title=paper.title,
        authors=paper.authors,
        chunk_count=chunk_count,
    )
