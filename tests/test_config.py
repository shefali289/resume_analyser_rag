import os
import unittest
from unittest.mock import patch

from resume_analyser.config import AppConfig


class ConfigTests(unittest.TestCase):
    def test_vercel_environment_variables_make_deployment_ready(self) -> None:
        environment = {
            "HF_API_KEY": "hf-secret",
            "SUPABASE_URL": "https://project.supabase.co/rest/v1",
            "SUPABASE_SERVICE_ROLE_KEY": "supabase-secret",
            "GEMINI_API_KEY": "gemini-secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = AppConfig.from_sources()

        self.assertTrue(config.deployment_ready)
        self.assertEqual(config.supabase_url, "https://project.supabase.co")
        self.assertEqual(config.gemini_model, "gemini-3.7-flash")

    def test_compatibility_aliases_are_supported(self) -> None:
        environment = {
            "HUGGINGFACEHUB_API_TOKEN": "hf-alias",
            "SUPABASE_URL": "https://project.supabase.co",
            "SUPABASE_KEY": "supabase-alias",
            "GOOGLE_API_KEY": "google-alias",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = AppConfig.from_sources()

        self.assertTrue(config.deployment_ready)
        self.assertEqual(config.hf_api_key, "hf-alias")
        self.assertEqual(config.supabase_key, "supabase-alias")
        self.assertEqual(config.gemini_api_key, "google-alias")

    def test_missing_variables_report_names_not_values(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = AppConfig.from_sources()

        self.assertEqual(
            config.missing_deployment_variables,
            (
                "HF_API_KEY",
                "SUPABASE_URL",
                "SUPABASE_SERVICE_ROLE_KEY",
                "GEMINI_API_KEY",
            ),
        )


if __name__ == "__main__":
    unittest.main()
