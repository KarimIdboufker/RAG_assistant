from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    anthropic_api_key: str
    openai_api_key: str

    pdf_dir: str = "pdfs"
    chunk_max_tokens: int = 400
    retrieval_top_k: int = 8

    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    claude_model: str = "claude-sonnet-4-6"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
