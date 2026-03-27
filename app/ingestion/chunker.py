"""
Structure-aware chunker — simplified.

Input:  the dict produced by extractor.extract_paper()
Output: list of chunk dicts ready for embedding + DB insert

Chunk types:
  "abstract" — the full abstract, one chunk
  "body"     — a paragraph or group of paragraphs within a section

Each chunk's "contextualized_content" prepends the paper title and section
heading so the embedding is self-contained (no need for the surrounding doc).
"""


def build_chunks(paper: dict, max_tokens: int = 400) -> list[dict]:
    """
    Returns:
      [
        {
          "chunk_type": str,
          "section": str,
          "chunk_index": int,
          "content": str,
          "contextualized_content": str,
          "page_num": int,          # always 0 — page tracking removed for simplicity
        },
        ...
      ]
    """
    title = paper["meta"].get("title") or "Unknown Paper"
    chunks: list[dict] = []

    # ── Abstract ─────────────────────────────────────────────────────────────
    if paper.get("abstract"):
        chunks.append(_make_chunk(
            chunk_type="abstract",
            content=paper["abstract"],
            section="Abstract",
            title=title,
        ))

    # ── Body sections ─────────────────────────────────────────────────────────
    for section in paper.get("sections", []):
        heading = section["heading"]
        paragraphs = [p.strip() for p in section["paragraphs"] if p.strip()]

        buffer: list[str] = []

        for para in paragraphs:
            buffer.append(para)

            if _word_count(buffer) >= max_tokens:
                # Flush everything except the last paragraph to avoid tiny leftover
                flush, buffer = buffer[:-1], buffer[-1:]
                if flush:
                    chunks.append(_make_chunk(
                        chunk_type="body",
                        content="\n\n".join(flush),
                        section=heading,
                        title=title,
                    ))

        if buffer:
            chunks.append(_make_chunk(
                chunk_type="body",
                content="\n\n".join(buffer),
                section=heading,
                title=title,
            ))

    # Assign sequential indices
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _word_count(paragraphs: list[str]) -> int:
    return sum(len(p.split()) for p in paragraphs)


def _make_chunk(chunk_type: str, content: str, section: str, title: str) -> dict:
    return {
        "chunk_type": chunk_type,
        "section": section,
        "chunk_index": 0,
        "content": content,
        "contextualized_content": (
            f"[Paper: {title}]\n[Section: {section}]\n\n{content}"
        ),
        "page_num": 0,
    }
