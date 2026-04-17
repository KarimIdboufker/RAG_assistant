"""
PDF text extractor — simplified to plain text only.

Extracts:
  - Metadata: title, authors (from PDF properties or page-1 heuristics)
  - Abstract: detected by keyword search in early pages
  - Pages: clean text per page

Returns:
  {
    "meta":     {"title": str|None, "authors": str|None, "filename": str},
    "abstract": str|None,
    "pages":    [{"text": str, "page_num": int}, ...]
  }
"""

import re
from collections import Counter

import fitz


_STOP_SECTIONS = {"references", "bibliography", "works cited", "acknowledgements", "acknowledgments"}


def extract_paper(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    body_size = _detect_body_font_size(doc)
    meta = _extract_metadata(doc, body_size)
    pages, abstract = _extract_pages(doc)
    doc.close()
    return {"meta": meta, "abstract": abstract, "pages": pages}


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────

def _extract_metadata(doc: fitz.Document, body_size: float) -> dict:
    pdf_meta = doc.metadata or {}
    title = (pdf_meta.get("title") or "").strip() or None
    authors = (pdf_meta.get("author") or "").strip() or None

    if (not title or not authors) and len(doc) > 0:
        title_h, authors_h = _heuristic_metadata(doc[0], body_size)
        title = title or title_h
        authors = authors or authors_h

    return {"title": title, "authors": authors, "filename": doc.name}


def _heuristic_metadata(page: fitz.Page, body_size: float) -> tuple[str | None, str | None]:
    candidates: list[tuple[float, str]] = []
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
            candidates.append((sum(sizes) / len(sizes), " ".join(texts)))

    candidates.sort(key=lambda x: x[0], reverse=True)
    title, authors = None, None
    for size, text in candidates[:6]:
        if size < body_size + 1:
            break
        if title is None:
            title = text
        elif authors is None and _looks_like_authors(text):
            authors = text
    return title, authors


def _looks_like_authors(text: str) -> bool:
    return "," in text or " and " in text.lower() or any(c.isdigit() for c in text)


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
# Page extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pages(doc: fitz.Document) -> tuple[list[dict], str | None]:
    pages = []
    abstract: str | None = None
    abstract_lines: list[str] = []
    in_abstract = False

    for page_num, page in enumerate(doc):
        raw = page.get_text("text")
        if _hits_stop_section(raw):   # check raw — lines are still intact
            break
        cleaned = _clean_text(raw)
        if not cleaned:
            continue

        # Detect and extract abstract from early pages
        if abstract is None and page_num < 3:
            abstract, in_abstract, abstract_lines = _try_extract_abstract(
                cleaned, in_abstract, abstract_lines
            )

        pages.append({"text": cleaned, "page_num": page_num + 1})

    if abstract is None and abstract_lines:
        abstract = " ".join(abstract_lines).strip() or None

    return pages, abstract


def _clean_text(text: str) -> str:
    text = text.replace("\x00", "")  # strip NUL chars — Postgres rejects them
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"[\d\s\-–—]+", line):  # page numbers, rules
            continue
        if len(line) < 3:
            continue
        lines.append(line)
    return " ".join(lines)


def _hits_stop_section(text: str) -> bool:
    # Check each original line, not just the joined string
    for line in text.splitlines():
        line = line.strip().lower().rstrip(".")
        if line in _STOP_SECTIONS:
            return True
    return False


def _try_extract_abstract(
    text: str,
    in_abstract: bool,
    lines: list[str],
) -> tuple[str | None, bool, list[str]]:
    lower = text.lower()

    if not in_abstract and "abstract" in lower:
        idx = lower.index("abstract")
        after = text[idx + len("abstract"):].strip().lstrip("—:-").strip()
        if len(after) > 80:
            return after[:3000], False, []
        in_abstract = True
        if after:
            lines.append(after)
        return None, True, lines

    if in_abstract:
        # Stop collecting when we hit a numbered section heading
        if re.match(r"^1[\.\s]", text.strip()):
            return " ".join(lines).strip() or None, False, lines
        lines.append(text[:2000])
        if len(" ".join(lines)) > 2000:
            return " ".join(lines).strip(), False, []

    return None, in_abstract, lines
