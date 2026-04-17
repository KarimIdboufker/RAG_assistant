"""
Simple sliding-window chunker.

Splits pages into overlapping word-count chunks.
No section detection — pure text only.
"""

_OVERLAP_WORDS = 50     # words carried over from previous chunk to next
_MIN_CHUNK_WORDS = 30   # discard chunks shorter than this


def build_chunks(paper: dict, max_tokens: int = 400) -> list[dict]:
    title = paper["meta"].get("title") or "Unknown Paper"
    chunks: list[dict] = []

    # ── Abstract ─────────────────────────────────────────────────────────────
    if paper.get("abstract"):
        chunks.append(_make_chunk(
            content=paper["abstract"],
            chunk_type="abstract",
            title=title,
            page_num=1,
        ))

    # ── Body — sliding window over all pages ──────────────────────────────────
    all_words: list[tuple[str, int]] = []  # (word, page_num)
    for page in paper.get("pages", []):
        for word in page["text"].split():
            all_words.append((word, page["page_num"]))

    i = 0
    while i < len(all_words):
        window = all_words[i: i + max_tokens]
        if len(window) < _MIN_CHUNK_WORDS:
            break

        content = " ".join(w for w, _ in window)
        page_num = window[0][1]

        chunks.append(_make_chunk(
            content=content,
            chunk_type="body",
            title=title,
            page_num=page_num,
        ))

        i += max_tokens - _OVERLAP_WORDS

    for idx, chunk in enumerate(chunks):
        chunk["chunk_index"] = idx

    return chunks


def _make_chunk(content: str, chunk_type: str, title: str, page_num: int) -> dict:
    return {
        "chunk_type": chunk_type,
        "section": "",
        "chunk_index": 0,
        "content": content,
        "contextualized_content": f"[Paper: {title}]\n\n{content}",
        "page_num": page_num,
    }
