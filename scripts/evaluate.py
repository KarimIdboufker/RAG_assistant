#!/usr/bin/env python3
"""
RAG evaluation — Precision@5, MRR, Hit@5.

Two test types:
  1. General  : "What is the main objective of this paper?"
                Run filtered per paper — measures within-paper retrieval quality.
  2. Specific : One unique question per paper, run against the full corpus.
                Measures whether retrieval focuses on the correct paper.
                Relevance = chunk comes from the expected paper.

Usage (from project root):
    python scripts/evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from app.config import settings
from app.database import SessionLocal
from app.embedding.embedder import embed_texts

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded questions
# ─────────────────────────────────────────────────────────────────────────────

GENERAL_QUESTION = "What is the main objective of this paper?"

SPECIFIC_QUESTIONS = {
    "2602.03300v1.pdf": "What is Collective Adversarial Data Synthesis?",
    "2602.04019v2.pdf": "How are PEFT and projected residual correction linked?",
    "2602.16977v1.pdf": "How to implement refusal mechanisms so that safety remains effective against prompt-based jailbreaks?",
    "2602.20457v1.pdf": "What is Self-Improving Efficient Online Alignment?",
    "2602.23200v1.pdf": "How to reduce hardware footprint during decoding of KV caching?",
}

K = 5


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(question: str, db, paper_id: str | None = None) -> list[dict]:
    prefixed = f"[Query]\n\n{question}"
    q_emb = embed_texts([prefixed])[0]
    emb_str = "[" + ",".join(map(str, q_emb)) + "]"

    paper_filter = "AND c.paper_id = :pid" if paper_id else ""
    params = {"emb": emb_str, "k": K}
    if paper_id:
        params["pid"] = paper_id

    rows = db.execute(
        text(f"""
            SELECT
                c.id::text     AS chunk_id,
                c.content,
                c.page_num,
                p.id::text     AS paper_id,
                p.filename,
                p.title,
                1 - (c.embedding <=> cast(:emb as vector)) AS score
            FROM chunks c
            JOIN papers p ON c.paper_id = p.id
            WHERE c.embedding IS NOT NULL
            {paper_filter}
            ORDER BY c.embedding <=> cast(:emb as vector)
            LIMIT :k
        """),
        params,
    ).fetchall()

    return [
        {
            "chunk_id":  r.chunk_id,
            "paper_id":  r.paper_id,
            "filename":  r.filename,
            "title":     r.title,
            "score":     float(r.score),
            "snippet":   r.content[:120].replace("\n", " "),
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (paper-level relevance)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], correct_paper_id: str) -> dict:
    hits = [r for r in results if r["paper_id"] == correct_paper_id]
    hit = len(hits) > 0

    rank = None
    for i, r in enumerate(results, 1):
        if r["paper_id"] == correct_paper_id:
            rank = i
            break

    precision = len(hits) / K
    mrr       = (1 / rank) if rank else 0.0
    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0

    return {
        "hit":       hit,
        "rank":      rank,
        "precision": precision,
        "mrr":       mrr,
        "avg_score": avg_score,
        "n_correct": len(hits),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sep(char="─", width=72): print(char * width)

def avg(metrics: list[dict], key: str) -> float:
    return sum(m[key] for m in metrics) / len(metrics) if metrics else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    db = SessionLocal()

    # Load paper records
    papers = {}
    for fname in SPECIFIC_QUESTIONS:
        row = db.execute(
            text("SELECT id::text, title, authors FROM papers WHERE filename = :f"),
            {"f": fname},
        ).fetchone()
        if not row:
            print(f"  NOT INGESTED: {fname}")
        else:
            papers[fname] = {"id": row.id, "title": row.title, "authors": row.authors}

    if len(papers) < len(SPECIFIC_QUESTIONS):
        missing = set(SPECIFIC_QUESTIONS) - set(papers)
        print(f"\nMissing papers — run ingest_all.py first:\n  " + "\n  ".join(missing))
        sys.exit(1)

    total_chunks = db.execute(text("SELECT COUNT(*) FROM chunks")).scalar()
    print(f"\nRAG Evaluation   k={K}   {len(papers)} papers   {total_chunks} total chunks")

    # ── 1. General question (per-paper, filtered) ─────────────────────────────
    sep("═")
    print(f"GENERAL QUESTION (filtered per paper)\nQ: \"{GENERAL_QUESTION}\"\n")
    sep()

    general_metrics = []
    for fname, paper in papers.items():
        results = retrieve(GENERAL_QUESTION, db, paper_id=paper["id"])
        m = compute_metrics(results, paper["id"])
        general_metrics.append(m)

        print(f"  {fname}  |  {(paper['title'] or '')[:50]}")
        print(f"  Chunks from this paper in top-{K}: {m['n_correct']}/{K}  "
              f"avg_score={m['avg_score']:.3f}")
        if results:
            print(f"  Top result: \"{results[0]['snippet']}...\"")
        print()

    print(f"  General avg_score : {avg(general_metrics, 'avg_score'):.3f}")
    print(f"  General P@{K}      : {avg(general_metrics, 'precision'):.3f}")

    # ── 2. Specific questions (full corpus) ───────────────────────────────────
    sep("═")
    print(f"SPECIFIC QUESTIONS (full corpus, {total_chunks} chunks)\n")
    sep()

    specific_metrics = []
    for fname, question in SPECIFIC_QUESTIONS.items():
        paper = papers[fname]
        results = retrieve(question, db)
        m = compute_metrics(results, paper["id"])
        specific_metrics.append(m)

        hit_str = f"rank={m['rank']}" if m["hit"] else "MISS"
        print(f"  {fname}")
        print(f"  Q  : {question}")
        print(f"  {hit_str}  P@{K}={m['precision']:.2f}  MRR={m['mrr']:.2f}  "
              f"avg_score={m['avg_score']:.3f}")
        print(f"  Top-{K} sources: " +
              "  ".join(f"{r['filename'][:20]}({r['score']:.2f})" for r in results))
        print()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    sep("═")
    print("AGGREGATE — Specific questions\n")
    print(f"  Precision@{K} : {avg(specific_metrics, 'precision'):.3f}")
    print(f"  MRR          : {avg(specific_metrics, 'mrr'):.3f}")
    print(f"  Hit@{K}       : {sum(m['hit'] for m in specific_metrics)}/{len(specific_metrics)}")
    print(f"  Avg score    : {avg(specific_metrics, 'avg_score'):.3f}")

    sep()
    print("\nPER-PAPER  (specific questions)\n")
    print(f"  {'File':<25} {'Hit':>4} {'Rank':>5} {'P@5':>5} {'MRR':>5} {'AvgScore':>9}")
    sep("-", 65)
    for (fname, _), m in zip(SPECIFIC_QUESTIONS.items(), specific_metrics):
        rank_str = str(m["rank"]) if m["rank"] else "—"
        print(f"  {fname:<25} {'✓' if m['hit'] else '✗':>4} {rank_str:>5} "
              f"{m['precision']:>5.2f} {m['mrr']:>5.2f} {m['avg_score']:>9.3f}")

    db.close()


if __name__ == "__main__":
    main()
