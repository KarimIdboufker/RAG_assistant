from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    anthropic_api_key: str

    pdf_dir: str = "pdfs"
    chunk_max_tokens: int = 400
    retrieval_top_k: int = 8

    # allenai-specter2 → 768 dims  (academic, recommended for scientific papers)
    # all-MiniLM-L6-v2 → 384 dims  (faster, use for local testing)
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    claude_model: str = "claude-sonnet-4-6"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
