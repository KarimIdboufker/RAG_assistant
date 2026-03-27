"""
Embedding via sentence-transformers (local, no API key required).

Default model: allenai-specter2  — trained on scientific papers, 768 dims.
Swap via EMBEDDING_MODEL in .env, e.g.:
  EMBEDDING_MODEL=all-MiniLM-L6-v2   (faster, 384 dims, set EMBEDDING_DIM=384)
"""

from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = _get_model().encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return [v.tolist() for v in vectors]
