"""Application configuration loaded from environment variables or Streamlit secrets."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration for external AI and database services."""

    hf_api_key: str = ""
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.7-flash"
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_table: str = "documents"
    supabase_query: str = "match_documents"

    @classmethod
    def from_sources(cls, secrets: Mapping[str, Any] | None = None) -> "AppConfig":
        """Load configuration, preferring Streamlit secrets over the environment."""

        secret_values = secrets or {}

        def value(name: str, default: str = "") -> str:
            candidate = secret_values.get(name, os.getenv(name, default))
            return str(candidate).strip() if candidate is not None else default

        def first_value(*names: str, default: str = "") -> str:
            """Return the first configured alias without exposing its value."""

            for name in names:
                candidate = value(name)
                if candidate:
                    return candidate
            return default

        supabase_url = value("SUPABASE_URL").rstrip("/")
        if supabase_url.endswith("/rest/v1"):
            supabase_url = supabase_url.removesuffix("/rest/v1")

        return cls(
            hf_api_key=first_value("HF_API_KEY", "HUGGINGFACEHUB_API_TOKEN"),
            hf_embedding_model=value(
                "HF_EMBEDDING_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            gemini_api_key=first_value("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            gemini_model=value("GEMINI_MODEL", "gemini-3.7-flash"),
            supabase_url=supabase_url,
            supabase_key=first_value("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_KEY"),
            supabase_table=value("SUPABASE_TABLE", "documents"),
            supabase_query=value("SUPABASE_QUERY", "match_documents"),
        )

    @property
    def semantic_search_ready(self) -> bool:
        return bool(self.hf_api_key)

    @property
    def ai_ready(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def supabase_ready(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)

    @property
    def vector_db_ready(self) -> bool:
        """Semantic retrieval always requires Hugging Face and Supabase."""

        return self.semantic_search_ready and self.supabase_ready

    @property
    def missing_vector_variables(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.hf_api_key:
            missing.append("HF_API_KEY")
        if not self.supabase_url:
            missing.append("SUPABASE_URL")
        if not self.supabase_key:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        return tuple(missing)

    @property
    def missing_deployment_variables(self) -> tuple[str, ...]:
        missing = list(self.missing_vector_variables)
        if not self.gemini_api_key:
            missing.append("GEMINI_API_KEY")
        return tuple(missing)

    @property
    def deployment_ready(self) -> bool:
        return not self.missing_deployment_variables
