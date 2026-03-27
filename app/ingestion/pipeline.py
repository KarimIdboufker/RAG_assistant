"""
Ingestion pipeline for a single PDF:
  1. Extract structure  (extractor.py  → metadata, abstract, sections)
  2. Build chunks       (chunker.py    → body paragraphs per section)
  3. Embed chunks       (embedder.py   → sentence-transformers, local)
  4. Persist to DB
"""

from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.embedding.embedder import embed_texts
from app.ingestion.chunker import build_chunks
from app.ingestion.extractor import extract_paper
from app.models import Chunk, Paper


def ingest_pdf(pdf_path: str, db: Session) -> tuple[Paper, int]:
    """
    Ingest a single PDF.  Skips files already present in the DB.

    Returns:
        (Paper record, number of chunks created)
        chunk count == 0 means the file was already ingested (skipped).
    """
    filename = Path(pdf_path).name

    existing = db.query(Paper).filter_by(filename=filename).first()
    if existing:
        count = db.query(Chunk).filter_by(paper_id=existing.id).count()
        return existing, 0  # 0 signals "skipped"

    # ── 1. Extract ──────────────────────────────────────────────────────────
    paper_data = extract_paper(pdf_path)
    meta = paper_data["meta"]

    # ── 2. Chunk ─────────────────────────────────────────────────────────────
    chunks = build_chunks(paper_data, max_tokens=settings.chunk_max_tokens)

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    texts = [c["contextualized_content"] for c in chunks]
    embeddings = embed_texts(texts)

    # ── 4. Persist ────────────────────────────────────────────────────────────
    paper = Paper(
        filename=filename,
        title=meta.get("title"),
        authors=meta.get("authors"),
    )
    db.add(paper)
    db.flush()  # get paper.id

    db_chunks = [
        Chunk(
            paper_id=paper.id,
            chunk_type=c["chunk_type"],
            section=c["section"],
            chunk_index=c["chunk_index"],
            content=c["content"],
            contextualized_content=c["contextualized_content"],
            embedding=embeddings[i],
            page_num=c["page_num"],
        )
        for i, c in enumerate(chunks)
    ]

    db.add_all(db_chunks)
    db.commit()

    return paper, len(db_chunks)
