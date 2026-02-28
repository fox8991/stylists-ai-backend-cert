"""Project configuration loaded from environment variables."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Project settings from environment variables."""

    OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5.2")
    QDRANT_URL: str | None = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY")
    TAVILY_API_KEY: str | None = os.getenv("TAVILY_API_KEY")


settings = Settings()
