#!/usr/bin/env python3
"""
Quick end-to-end test: ingest one PDF and run a similarity query.

Usage (from project root, with DB running):
    python scripts/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.database import SessionLocal, init_db, engine
from app.ingestion.pipeline import ingest_pdf
from app.embedding.embedder import embed_texts
from sqlalchemy import text

PDF = "pdfs/2602.03300v1.pdf"
TEST_QUERY = "What is the main contribution of this paper?"


def main():
    # ── 1. Init DB ────────────────────────────────────────────────────────────
    print("Initialising database …")
    init_db()
    print("  OK\n")

    # ── 2. Ingest ─────────────────────────────────────────────────────────────
    pdf_path = Path(PDF)
    if not pdf_path.exists():
        print(f"ERROR: {PDF} not found. Run from project root.")
        sys.exit(1)

    db = SessionLocal()
    try:
        print(f"Ingesting {pdf_path.name} …")
        paper, n_chunks = ingest_pdf(str(pdf_path), db)

        if n_chunks == 0:
            print(f"  Already ingested (skipped). Paper: {paper.title}")
        else:
            print(f"  OK — {n_chunks} chunks created")
            print(f"  Title:   {paper.title or '(none)'}")
            print(f"  Authors: {paper.authors or '(none)'}")

        # ── 3. Chunk breakdown ────────────────────────────────────────────────
        rows = db.execute(
            text("""
                SELECT chunk_type, section, chunk_index, length(content) AS chars
                FROM chunks
                WHERE paper_id = :pid
                ORDER BY chunk_index
            """),
            {"pid": str(paper.id)},
        ).fetchall()

        print(f"\n── Chunk breakdown ({len(rows)} total) ──────────────────────")
        for r in rows:
            print(f"  [{r.chunk_index:>3}] {r.chunk_type:<10} | {(r.section or '')[:50]:<50} | {r.chars} chars")

        # ── 4. Similarity query ───────────────────────────────────────────────
        print(f"\n── Similarity query ─────────────────────────────────────────")
        print(f"  Q: {TEST_QUERY}\n")

        q_emb = embed_texts([TEST_QUERY])[0]
        emb_str = "[" + ",".join(map(str, q_emb)) + "]"

        hits = db.execute(
            text("""
                SELECT
                    c.chunk_index,
                    c.chunk_type,
                    c.section,
                    c.page_num,
                    1 - (c.embedding <=> cast(:emb as vector)) AS score,
                    left(c.content, 200)                AS snippet
                FROM chunks c
                WHERE c.paper_id = :pid
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> cast(:emb as vector)
                LIMIT 5
            """),
            {"emb": emb_str, "pid": str(paper.id)},
        ).fetchall()

        for i, h in enumerate(hits, 1):
            print(f"  #{i}  score={h.score:.4f}  [{h.chunk_type}] {h.section or ''}")
            print(f"      {h.snippet.strip()[:180]} …")
            print()

    finally:
        db.close()

    print("Done.")


if __name__ == "__main__":
    main()
