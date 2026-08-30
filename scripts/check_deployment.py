"""Validate local or Vercel configuration without printing secret values.

Usage:
    python scripts/check_deployment.py
    python scripts/check_deployment.py --live

The live mode sends only a harmless health-check string to providers. It never
loads or transmits a resume and it never inserts rows into Supabase.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.parse import urlparse
import warnings

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from resume_analyser.config import AppConfig  # noqa: E402


EXPECTED_EMBEDDING_DIMENSIONS = 384


def _safe_error(error: Exception, config: AppConfig) -> str:
    """Remove every configured credential from provider error messages."""

    message = str(error)
    for secret in (
        config.hf_api_key,
        config.gemini_api_key,
        config.supabase_key,
    ):
        if secret:
            message = message.replace(secret, "***")
    return f"{type(error).__name__}: {message}"


def _check_static_config(config: AppConfig) -> list[str]:
    errors: list[str] = []
    if config.missing_deployment_variables:
        errors.append(
            "Missing environment variable(s): "
            + ", ".join(config.missing_deployment_variables),
        )

    parsed_url = urlparse(config.supabase_url)
    if config.supabase_url and (
        parsed_url.scheme != "https" or not parsed_url.netloc
    ):
        errors.append("SUPABASE_URL must be a valid HTTPS project base URL.")
    if config.supabase_url.endswith("/rest/v1"):
        errors.append("SUPABASE_URL must not include /rest/v1.")
    if not config.supabase_table:
        errors.append("SUPABASE_TABLE cannot be empty.")
    if not config.supabase_query:
        errors.append("SUPABASE_QUERY cannot be empty.")
    return errors


def _check_live_services(config: AppConfig) -> list[str]:
    errors: list[str] = []

    try:
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        embeddings = HuggingFaceEndpointEmbeddings(
            model=config.hf_embedding_model,
            huggingfacehub_api_token=config.hf_api_key,
            provider="hf-inference",
        )
        vector = embeddings.embed_query("CareerLens deployment health check")
        if len(vector) != EXPECTED_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding dimension is {len(vector)}, expected "
                f"{EXPECTED_EMBEDDING_DIMENSIONS} for supabase/schema.sql.",
            )
        print(f"PASS  Hugging Face ({len(vector)} dimensions)")
    except Exception as error:
        errors.append("Hugging Face: " + _safe_error(error, config))
        vector = []

    if vector:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                from langchain_community.vectorstores import SupabaseVectorStore
            from supabase import create_client

            vector_store = SupabaseVectorStore(
                client=create_client(config.supabase_url, config.supabase_key),
                embedding=embeddings,
                table_name=config.supabase_table,
                query_name=config.supabase_query,
            )
            # A read-only RPC verifies the table/function/key combination.
            vector_store.similarity_search_by_vector(
                vector,
                k=1,
                filter={"resume_id": "__deployment_health_check__"},
            )
            print("PASS  Supabase vector search RPC")
        except Exception as error:
            errors.append("Supabase: " + _safe_error(error, config))

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
            # Gemini 3.x may spend some output tokens on internal reasoning.
            max_output_tokens=256,
            max_retries=1,
        )
        response = model.invoke("Reply with the single word OK.")
        if not response.text.strip():
            raise RuntimeError("Gemini returned an empty health-check response.")
        print(f"PASS  Gemini ({config.gemini_model})")
    except Exception as error:
        errors.append("Gemini: " + _safe_error(error, config))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call Hugging Face, Supabase, and Gemini with harmless test data.",
    )
    arguments = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    config = AppConfig.from_sources()
    errors = _check_static_config(config)

    if not errors:
        print("PASS  Required environment variables are configured")
        print(f"INFO  Embedding model: {config.hf_embedding_model}")
        print(f"INFO  Gemini model: {config.gemini_model}")
        print(
            f"INFO  Supabase target: {config.supabase_table} / "
            f"{config.supabase_query}",
        )
        if arguments.live:
            errors.extend(_check_live_services(config))

    if errors:
        for error in errors:
            print(f"FAIL  {error}", file=sys.stderr)
        return 1

    print("Deployment preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
