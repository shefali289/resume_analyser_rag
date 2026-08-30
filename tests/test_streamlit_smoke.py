import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

from resume_analyser.parsers import ParsedResume


class _FakeSupabaseClient:
    """Answers the Engineer panel's row count without touching the network."""

    def __init__(self, count: int) -> None:
        self._count = count

    def table(self, _table_name):
        return self

    def select(self, _columns, count=None):
        return self

    def contains(self, _column, _value):
        return self

    def execute(self):
        return SimpleNamespace(count=self._count, data=None)


SUPABASE_ENV = {
    "HF_API_KEY": "test-hf-key",
    "GEMINI_API_KEY": "test-gemini-key",
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-key",
}


def _sample_resume() -> ParsedResume:
    return ParsedResume(
        filename="portfolio-resume.txt",
        text=(
            "SUMMARY\nData engineer\nSKILLS\nPython SQL AWS\nEXPERIENCE\n"
            + "- Built reliable analytics pipelines and improved delivery by 30%.\n" * 8
            + "EDUCATION\nBachelor degree"
        ),
        file_type="TXT",
        page_count=1,
        word_count=75,
        resume_id="test-resume",
    )


def _app_path() -> Path:
    return Path(__file__).resolve().parents[1] / "streamlit_app.py"


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class StreamlitSmokeTests(unittest.TestCase):
    def test_landing_page_renders_without_exceptions(self) -> None:
        app = AppTest.from_file(_app_path(), default_timeout=15).run()

        self.assertEqual(list(app.exception), [])
        self.assertGreaterEqual(len(app.file_uploader), 1)
        self.assertTrue(any("Upload a TXT" in item.value for item in app.info))

    def test_resume_dashboard_renders_without_external_calls(self) -> None:
        app = AppTest.from_file(_app_path(), default_timeout=15).run()
        app.session_state["resume"] = _sample_resume()
        app.session_state["resume_signature"] = "test-signature"
        app.run()

        self.assertEqual(list(app.exception), [])
        self.assertEqual(len(app.tabs), 2)
        self.assertTrue(any("Loaded portfolio-resume.txt" in item.value for item in app.success))
        self.assertEqual(app.radio[0].value, "General")
        self.assertFalse(
            any("Engineer pipeline" in item.value for item in app.markdown),
        )

        with (
            patch.dict("os.environ", SUPABASE_ENV, clear=False),
            patch("supabase.create_client", return_value=_FakeSupabaseClient(0)),
        ):
            app.radio[0].set_value("Engineer").run()

        self.assertEqual(list(app.exception), [])
        self.assertTrue(
            any("Engineer pipeline" in item.value for item in app.markdown),
        )


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class EngineerIndexStatusTests(unittest.TestCase):
    """The Engineer card must reflect Supabase, not just this session."""

    def _run_engineer_mode(self, stored_chunks: int):
        client = _FakeSupabaseClient(stored_chunks)
        with (
            patch.dict("os.environ", SUPABASE_ENV, clear=False),
            patch("supabase.create_client", return_value=client),
        ):
            app = AppTest.from_file(_app_path(), default_timeout=15).run()
            app.session_state["resume"] = _sample_resume()
            app.session_state["resume_signature"] = "test-signature"
            app.radio[0].set_value("Engineer").run()
        return app

    def test_unindexed_resume_offers_an_explicit_index_action(self) -> None:
        app = self._run_engineer_mode(stored_chunks=0)

        self.assertEqual(list(app.exception), [])
        labels = [button.label for button in app.button]
        self.assertIn("Index resume", labels)
        self.assertIn("Refresh status", labels)
        self.assertTrue(any("Not indexed" in item.value for item in app.markdown))

    def test_indexing_is_blocked_until_external_ai_is_enabled(self) -> None:
        """Indexing ships resume text outward, so it needs the consent toggle."""

        app = self._run_engineer_mode(stored_chunks=0)
        index_button = next(
            button for button in app.button if button.label == "Index resume"
        )

        self.assertEqual(list(app.exception), [])
        self.assertTrue(index_button.disabled)
        self.assertTrue(
            any(
                "Enable RAG + GenAI in the sidebar to index" in item.value
                for item in app.caption
            ),
        )

    def test_status_reports_chunks_already_stored_in_supabase(self) -> None:
        app = self._run_engineer_mode(stored_chunks=12)

        self.assertEqual(list(app.exception), [])
        self.assertIn("Re-index resume", [button.label for button in app.button])
        self.assertTrue(
            any(
                "Supabase semantic" in item.value and "12 chunks" in item.value
                for item in app.markdown
            ),
        )

    def test_index_failure_is_reported_in_the_panel(self) -> None:
        client = _FakeSupabaseClient(0)
        with (
            patch.dict("os.environ", SUPABASE_ENV, clear=False),
            patch("supabase.create_client", return_value=client),
            patch(
                "resume_analyser.rag.HuggingFaceEndpointEmbeddings",
                side_effect=RuntimeError("embedding endpoint unavailable"),
            ),
        ):
            app = AppTest.from_file(_app_path(), default_timeout=15).run()
            app.session_state["resume"] = _sample_resume()
            app.session_state["resume_signature"] = "test-signature"
            app.radio[0].set_value("Engineer").run()
            enable = next(
                item for item in app.toggle if item.label == "Enable RAG + GenAI"
            )
            enable.set_value(True).run()
            index_button = next(
                button for button in app.button if button.label == "Index resume"
            )
            self.assertFalse(index_button.disabled)
            index_button.click().run()

        self.assertEqual(list(app.exception), [])
        self.assertTrue(
            any(
                "Indexing failed" in item.value
                and "embedding endpoint unavailable" in item.value
                for item in app.error
            ),
        )


if __name__ == "__main__":
    unittest.main()
