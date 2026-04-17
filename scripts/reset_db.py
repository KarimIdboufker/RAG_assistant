#!/usr/bin/env python3
"""
Clears all ingested data and recreates the schema.
Run this when changing embedding models or vector dimensions.

Usage (from project root):
    python scripts/reset_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal, engine, Base
from app.models import Paper, Chunk


def main():
    print("Clearing papers and chunks...")
    db = SessionLocal()
    try:
        chunks = db.query(Chunk).delete()
        papers = db.query(Paper).delete()
        db.commit()
        print(f"  Deleted {chunks} chunk(s) and {papers} paper(s).")
    finally:
        db.close()

    print("Recreating schema...")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    print("  Done — schema recreated.")


if __name__ == "__main__":
    main()
