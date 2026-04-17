from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

import anthropic

from app.config import settings
from app.database import get_db
from app.embedding.embedder import embed_texts

router = APIRouter()

_anthropic_client: anthropic.Anthropic | None = None

_MIN_SCORE = 0.25       # discard chunks below this cosine similarity
_MAX_PER_PAPER = 3      # cap chunks from any single paper


def _anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _anthropic_client


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 8


class Source(BaseModel):
    title: str
    section: str
    chunk_type: str
    content: str
    page_num: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/", response_model=QueryResponse)
def query(req: QueryRequest, db: Session = Depends(get_db)):
    # 1. Embed the question — prefix matches chunk contextualized_content format
    prefixed = f"[Query]\n\n{req.question}"
    q_emb = embed_texts([prefixed])[0]
    emb_str = "[" + ",".join(map(str, q_emb)) + "]"

    # 2. Vector similarity search — fetch more candidates than top_k to allow filtering
    fetch_k = req.top_k * 4
    rows = db.execute(
        text("""
            SELECT
                c.content,
                c.section,
                c.chunk_type,
                c.page_num,
                p.title,
                p.authors,
                p.id AS paper_id,
                1 - (c.embedding <=> cast(:emb as vector)) AS score
            FROM chunks c
            JOIN papers p ON c.paper_id = p.id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> cast(:emb as vector)
            LIMIT :k
        """),
        {"emb": emb_str, "k": fetch_k},
    ).fetchall()

    # 3. Filter by score threshold and cap per paper
    paper_counts: dict = {}
    filtered = []
    for row in rows:
        if row.score < _MIN_SCORE:
            continue
        pid = str(row.paper_id)
        if paper_counts.get(pid, 0) >= _MAX_PER_PAPER:
            continue
        paper_counts[pid] = paper_counts.get(pid, 0) + 1
        filtered.append(row)
        if len(filtered) >= req.top_k:
            break

    if not filtered:
        return QueryResponse(
            answer="No sufficiently relevant content found in the knowledge base.",
            sources=[],
        )

    # 4. Build context for Claude — include paper metadata as a header
    seen_titles = {}
    for row in filtered:
        if row.title and row.title not in seen_titles:
            seen_titles[row.title] = row.authors

    meta_block = "\n".join(
        f"- {title} (Authors: {authors or 'unknown'})"
        for title, authors in seen_titles.items()
    )

    context_blocks = [
        f"[{row.title or 'Unknown'} — {row.section or ''}]\n{row.content}"
        for row in filtered
    ]
    context = "\n\n---\n\n".join(context_blocks)

    sources = [
        Source(
            title=row.title or "Unknown",
            section=row.section or "",
            chunk_type=row.chunk_type,
            content=row.content,
            page_num=row.page_num or 0,
            score=round(float(row.score), 4),
        )
        for row in filtered
    ]

    # 5. Generate answer with Claude
    prompt = (
        f"Papers referenced below:\n{meta_block}\n\n"
        f"Context from academic papers:\n\n{context}\n\n"
        f"Question: {req.question}"
    )

    response = _anthropic().messages.create(
        model=settings.claude_model,
        max_tokens=2048,
        system=(
            "You are a research assistant specializing in academic papers. "
            "Answer the question using only the provided context excerpts. "
            "Cite specific papers and sections when making claims, e.g. "
            "(Smith et al. — Section 3.2). "
            "If the context does not contain enough information to answer "
            "the question, say so explicitly rather than speculating."
        ),
        messages=[{"role": "user", "content": prompt}],
    )

    answer = next(b.text for b in response.content if b.type == "text")

    return QueryResponse(answer=answer, sources=sources)
