#!/usr/bin/env python3
"""
Bulk ingestion script — run once to process all PDFs.

Usage (from project root):
    python scripts/ingest_all.py

Or inside Docker:
    docker compose exec app python scripts/ingest_all.py
"""

import sys
import os
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.database import SessionLocal, init_db
from app.ingestion.pipeline import ingest_pdf


def main():
    print("Initialising database …")
    init_db()

    pdf_dir = Path(settings.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in '{pdf_dir}'. Exiting.")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDFs in '{pdf_dir}'\n")

    db = SessionLocal()
    ok = err = skipped = 0

    try:
        for i, pdf_path in enumerate(pdfs, 1):
            label = f"[{i:>2}/{len(pdfs)}] {pdf_path.name}"
            try:
                paper, n_chunks = ingest_pdf(str(pdf_path), db)
                if n_chunks == 0:
                    print(f"  SKIP  {label}  (already ingested)")
                    skipped += 1
                else:
                    title = paper.title or "(no title)"
                    print(f"  OK    {label}  →  {n_chunks} chunks  |  {title}")
                    ok += 1
            except Exception as exc:
                print(f"  ERROR {label}  →  {exc}")
                db.rollback()
                err += 1
    finally:
        db.close()

    print(f"\nDone — {ok} ingested, {skipped} skipped, {err} errors.")


if __name__ == "__main__":
    main()
