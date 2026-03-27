"""
PDF structure extractor — simplified.

Extracts:
  - Metadata: title, authors (from PDF properties or page-1 heuristics)
  - Abstract: detected by keyword, kept as a single block
  - Sections: heading + body paragraphs, grouped together
  - Stops at the first of: References / Bibliography / Acknowledgements

Returns:
  {
    "meta":  {"title": str|None, "authors": str|None, "filename": str},
    "abstract": str|None,
    "sections": [{"heading": str, "paragraphs": [str, ...]}, ...]
  }
"""

from collections import Counter

import fitz  # pymupdf


# Section headings that mark the end of the main body
_STOP_HEADINGS = {
    "references", "bibliography", "works cited",
    "acknowledgements", "acknowledgments", "appendix",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_paper(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    body_size = _detect_body_font_size(doc)
    meta = _extract_metadata(doc, body_size)

    raw_sections = _extract_sections(doc, body_size)
    abstract, sections = _split_abstract(raw_sections)

    doc.close()
    return {
        "meta": meta,
        "abstract": abstract,
        "sections": sections,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metadata(doc: fitz.Document, body_size: float) -> dict:
    pdf_meta = doc.metadata or {}
    title = (pdf_meta.get("title") or "").strip() or None
    authors = (pdf_meta.get("author") or "").strip() or None

    # ArXiv PDFs almost always have empty metadata — extract from page 1
    if (not title or not authors) and len(doc) > 0:
        title_h, authors_h = _heuristic_metadata(doc[0], body_size)
        title = title or title_h
        authors = authors or authors_h

    return {"title": title, "authors": authors, "filename": doc.name}


def _heuristic_metadata(page: fitz.Page, body_size: float) -> tuple[str | None, str | None]:
    """
    On the first page of an academic paper:
      - Title   = the largest text block (significantly above body size)
      - Authors = the next-largest block (slightly above body or same size but
                  contains commas / numbers suggesting author list)
    """
    candidates: list[tuple[float, str]] = []  # (avg_size, text)

    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        sizes, texts = [], []
        for line in block["lines"]:
            for span in line["spans"]:
                t = span["text"].strip()
                if t:
                    sizes.append(span["size"])
                    texts.append(t)
        if sizes:
            avg = sum(sizes) / len(sizes)
            candidates.append((avg, " ".join(texts)))

    # Sort largest first
    candidates.sort(key=lambda x: x[0], reverse=True)

    title = None
    authors = None

    for i, (size, text) in enumerate(candidates[:6]):  # only inspect top blocks
        if size < body_size + 1:
            break
        if title is None:
            title = text
        elif authors is None and _looks_like_authors(text):
            authors = text

    return title, authors


def _looks_like_authors(text: str) -> bool:
    """Rough heuristic: author lines have commas, numbers (affiliations), or 'and'."""
    return (
        "," in text
        or " and " in text.lower()
        or any(c.isdigit() for c in text)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Font analysis
# ─────────────────────────────────────────────────────────────────────────────

def _detect_body_font_size(doc: fitz.Document) -> float:
    sizes: list[float] = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes.append(round(span["size"] * 2) / 2)
    return Counter(sizes).most_common(1)[0][0] if sizes else 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Section extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sections(doc: fitz.Document, body_size: float) -> list[dict]:
    """
    Walk all pages and build a flat list of sections:
      [{"heading": str, "paragraphs": [str, ...]}, ...]

    Stops when a stop-heading (References etc.) is encountered.
    """
    sections: list[dict] = []
    current: dict = {"heading": "Preamble", "paragraphs": []}
    stop = False

    for page in doc:
        if stop:
            break

        # Sort blocks top→bottom, left→right (handles two-column layouts roughly)
        blocks = sorted(
            page.get_text("dict")["blocks"],
            key=lambda b: (round(b["bbox"][1] / 20), b["bbox"][0]),
        )

        for block in blocks:
            if block["type"] != 0:  # skip image blocks
                continue

            block_text, is_heading = _classify_block(block, body_size)
            if not block_text:
                continue

            if is_heading:
                # Check for stop-section
                if block_text.strip().lower().rstrip(".") in _STOP_HEADINGS:
                    stop = True
                    break

                # Save previous section and start a new one
                if current["paragraphs"]:
                    sections.append(current)
                current = {"heading": block_text.strip(), "paragraphs": []}
            else:
                current["paragraphs"].append(block_text)

    if current["paragraphs"]:
        sections.append(current)

    return sections


def _classify_block(block: dict, body_size: float) -> tuple[str, bool]:
    """
    Returns (text, is_heading).
    A block is a heading if its dominant font is significantly larger or bold
    relative to the body baseline.
    """
    lines_text: list[str] = []
    heading_spans = 0
    total_spans = 0

    for line in block["lines"]:
        parts = []
        for span in line["spans"]:
            t = span["text"].strip()
            if not t:
                continue
            total_spans += 1
            size = span["size"]
            bold = bool(span["flags"] & 2**4)
            if size >= body_size + 2 or (bold and size >= body_size + 0.5):
                heading_spans += 1
            parts.append(span["text"])
        if parts:
            lines_text.append("".join(parts))

    text = "\n".join(lines_text).strip()
    is_heading = total_spans > 0 and (heading_spans / total_spans) >= 0.6

    return text, is_heading


# ─────────────────────────────────────────────────────────────────────────────
# Abstract separation
# ─────────────────────────────────────────────────────────────────────────────

def _split_abstract(sections: list[dict]) -> tuple[str | None, list[dict]]:
    """
    Find the abstract section (by heading keyword or position) and return it
    separately from the body sections.
    """
    abstract_text: str | None = None
    body_sections: list[dict] = []

    for section in sections:
        heading_lower = section["heading"].lower()
        if abstract_text is None and (
            "abstract" in heading_lower
            or heading_lower in ("preamble", "summary")
        ):
            abstract_text = "\n\n".join(section["paragraphs"]).strip() or None
        else:
            body_sections.append(section)

    return abstract_text, body_sections
