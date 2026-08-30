import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from resume_analyser.ai import AIRequest
from resume_analyser.config import AppConfig
from resume_analyser.rag import (
    ResumeRAGPipeline,
    count_stored_chunks,
    split_resume,
)


class _FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, _query):
        return self.documents


class _FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def as_retriever(self, **_kwargs):
        return _FakeRetriever(self.documents)


class _FakeQuery:
    """Records the read-only chain used to count rows for one resume."""

    def __init__(self, table, count):
        self.table = table
        self._count = count
        self.filter = None

    def select(self, _columns, count=None):
        self.table.select_count = count
        return self

    def contains(self, column, value):
        self.filter = (column, value)
        return self

    def execute(self):
        return SimpleNamespace(count=self._count, data=None)


class _FakeSupabaseClient:
    def __init__(self, count=7):
        self._count = count
        self.table_name = ""
        self.select_count = ""
        self.query = None

    def table(self, table_name):
        self.table_name = table_name
        self.query = _FakeQuery(self, self._count)
        return self.query


class StoredChunkCountTests(unittest.TestCase):
    def test_counts_only_rows_for_the_requested_resume(self) -> None:
        config = AppConfig(
            supabase_url="https://example.supabase.co",
            supabase_key="test-service-key",
            supabase_table="documents",
        )
        client = _FakeSupabaseClient(count=7)

        with patch("supabase.create_client", return_value=client):
            stored = count_stored_chunks(config, "resume-123")

        self.assertEqual(stored, 7)
        self.assertEqual(client.table_name, "documents")
        self.assertEqual(client.select_count, "exact")
        self.assertEqual(client.query.filter, ("metadata", {"resume_id": "resume-123"}))

    def test_requires_supabase_configuration_and_a_resume_id(self) -> None:
        configured = AppConfig(
            supabase_url="https://example.supabase.co",
            supabase_key="test-service-key",
        )

        with self.assertRaisesRegex(ValueError, "Supabase configuration"):
            count_stored_chunks(AppConfig(), "resume-123")
        with self.assertRaisesRegex(ValueError, "resume ID"):
            count_stored_chunks(configured, "   ")


class RetrievalTests(unittest.TestCase):
    def test_langchain_splitter_uses_overlap_metadata_and_stable_ids(self) -> None:
        first_run = split_resume(
            "A" * 1_200,
            resume_id="resume-123",
            filename="candidate.txt",
            chunk_size=500,
            chunk_overlap=50,
        )
        second_run = split_resume(
            "A" * 1_200,
            resume_id="resume-123",
            filename="candidate.txt",
            chunk_size=500,
            chunk_overlap=50,
        )

        self.assertEqual(len(first_run), 3)
        self.assertEqual(first_run[0].page_content[-50:], first_run[1].page_content[:50])
        self.assertEqual(first_run[0].metadata["resume_id"], "resume-123")
        self.assertEqual(first_run[1].metadata["chunk_index"], 1)
        self.assertEqual(first_run[0].id, second_run[0].id)
        self.assertTrue(first_run[0].id.isdigit())

    def test_pipeline_requires_vector_database_configuration(self) -> None:
        pipeline = ResumeRAGPipeline(AppConfig())

        with self.assertRaisesRegex(ValueError, "HF_API_KEY"):
            pipeline.prepare_resume(
                "Python engineer who built reliable data pipelines and APIs.",
                resume_id="resume-123",
                filename="candidate.txt",
            )

    def test_pipeline_always_uses_supabase_semantic_mode(self) -> None:
        config = AppConfig(
            hf_api_key="test-hf",
            supabase_url="https://example.supabase.co",
            supabase_key="test-service-key",
        )
        pipeline = ResumeRAGPipeline(config)

        with (
            patch.object(pipeline, "_create_embeddings", return_value=object()),
            patch.object(
                pipeline,
                "_create_vector_store",
                side_effect=lambda: _FakeVectorStore(pipeline.documents),
            ),
        ):
            chunk_count = pipeline.prepare_resume(
                "Python engineer who built reliable data pipelines and APIs.",
                resume_id="resume-123",
                filename="candidate.txt",
            )

        context = pipeline.build_context("Python pipelines")
        self.assertEqual(chunk_count, 1)
        self.assertEqual(pipeline.mode, "Supabase semantic")
        self.assertIn("Resume chunk 1", context)
        self.assertIn("data pipelines", context)

    def test_answer_uses_prompt_model_parser_chain(self) -> None:
        pipeline = ResumeRAGPipeline(
            AppConfig(gemini_api_key="test-key", gemini_model="test-model"),
        )
        fake_model = RunnableLambda(
            lambda prompt: AIMessage(content=f"Grounded: {prompt.to_string()[:12]}"),
        )

        with patch(
            "resume_analyser.rag.ChatGoogleGenerativeAI",
            return_value=fake_model,
        ):
            result = pipeline.answer(
                AIRequest(task="analysis", context="Built Python data pipelines."),
            )

        self.assertTrue(result.startswith("Grounded:"))


if __name__ == "__main__":
    unittest.main()
