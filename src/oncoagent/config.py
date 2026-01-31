"""Configuration helpers for OncoAgent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment."""

    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    exa_api_key: str | None = os.getenv("EXA_API_KEY")
    database_url: str | None = os.getenv("DATABASE_URL")


def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()

