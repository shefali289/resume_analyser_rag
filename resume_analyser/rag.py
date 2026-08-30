"""Easy-to-follow LangChain RAG pipeline for resume analysis.

The stages in :class:`ResumeRAGPipeline` deliberately mirror ``index.js``:

1. split the resume into documents;
2. create Hugging Face embeddings;
3. persist vectors in Supabase;
4. create a retriever and fetch relevant documents;
5. place the retrieved context in a prompt;
6. call Gemini and parse its text response.

Keeping the stages separate makes this module useful as both application code
and a readable learning example.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from resume_analyser.ai import AIRequest, build_career_prompt
from resume_analyser.config import AppConfig


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved resume chunk with display-friendly metadata."""

    index: int
    text: str
    score: float | None
    method: str


def count_stored_chunks(config: AppConfig, resume_id: str) -> int:
    """Return the actual number of Supabase rows stored for one resume.

    This is deliberately a metadata-only query: it does not download the
    embedding vectors or resume text.  It lets the Engineer UI distinguish a
    fresh local session from a resume that is already indexed remotely.
    """

    if not config.supabase_ready:
        raise ValueError("Supabase configuration is incomplete.")
    if not resume_id.strip():
        raise ValueError("A resume ID is required to check the index.")

    from supabase import create_client

    client = create_client(config.supabase_url, config.supabase_key)
    response = (
        client.table(config.supabase_table)
        .select("id", count="exact")
        .contains("metadata", {"resume_id": resume_id})
        .execute()
    )
    if response.count is not None:
        return int(response.count)
    return len(response.data or [])


def split_resume(
    text: str,
    *,
    resume_id: str = "",
    filename: str = "resume",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Stage 1: split resume text exactly like the splitter in ``index.js``.

    LangChain ``Document`` objects keep the text in ``page_content`` and attach
    metadata that follows every chunk through Supabase storage and retrieval.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = splitter.create_documents(
        [text],
        metadatas=[{"resume_id": resume_id, "filename": filename}],
    )

    # Store a readable chunk number in metadata for citations in the UI.
    for index, document in enumerate(documents):
        document.metadata["chunk_index"] = index
        # LangChain passes Document IDs to Supabase. A deterministic bigint ID
        # makes repeated serverless invocations upsert the same chunk instead of
        # creating duplicates. The sign bit is cleared for PostgreSQL bigint.
        digest = blake2b(f"{resume_id}:{index}".encode(), digest_size=8).digest()
        document.id = str(int.from_bytes(digest, "big") & ((1 << 63) - 1) or 1)
    return documents


class ResumeRAGPipeline:
    """The Streamlit app's RAG workflow, arranged like the Node.js prototype."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.documents: list[Document] = []
        self.embeddings: Any = None
        self.vector_store: Any = None
        self.retriever: Any = None
        self.mode = "Not built"

    def prepare_resume(
        self,
        text: str,
        *,
        resume_id: str,
        filename: str,
    ) -> int:
        """Run stages 1-4 and return the number of indexed chunks."""

        # Stage 1: turn the resume into overlapping LangChain Documents.
        self.documents = split_resume(
            text,
            resume_id=resume_id,
            filename=filename,
        )
        if not self.documents:
            raise ValueError("The resume contains no text to index.")

        if not self.config.vector_db_ready:
            missing = ", ".join(self.config.missing_vector_variables)
            raise ValueError(f"Vector database configuration is incomplete: {missing}")

        # Stage 2: use the hosted Hugging Face model from the original app.
        self.embeddings = self._create_embeddings()

        # Stage 3: embed and persist every chunk in Supabase. There is no
        # in-memory vector-store path, so deployed instances share one source.
        self.vector_store = self._create_vector_store()

        # Stage 4: expose the vector store through LangChain's retriever API.
        search_kwargs: dict[str, Any] = {
            "k": 6,
            "filter": {"resume_id": resume_id},
        }
        self.retriever = self.vector_store.as_retriever(
            search_kwargs=search_kwargs,
        )
        self.mode = "Supabase semantic"
        return len(self.documents)

    def _create_embeddings(self) -> HuggingFaceEndpointEmbeddings:
        """Stage 2: configure Hugging Face embeddings."""

        return HuggingFaceEndpointEmbeddings(
            model=self.config.hf_embedding_model,
            huggingfacehub_api_token=self.config.hf_api_key,
            provider="hf-inference",
        )

    def _create_vector_store(self) -> Any:
        """Stage 3: create and populate the required Supabase vector store."""

        # These integrations are imported here to keep the stage self-contained.
        from langchain_community.vectorstores import SupabaseVectorStore
        from supabase import create_client

        client = create_client(self.config.supabase_url, self.config.supabase_key)
        return SupabaseVectorStore.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            client=client,
            table_name=self.config.supabase_table,
            query_name=self.config.supabase_query,
        )

    def retrieve(self, query: str, *, limit: int = 6) -> list[RetrievalResult]:
        """Stage 5: retrieve the chunks that are most relevant to a question."""

        if not query.strip():
            return []
        if self.retriever is None:
            raise RuntimeError("Call prepare_resume() before retrieve().")

        # This is the Python equivalent of ``await retriever.invoke(query)``.
        documents = self.retriever.invoke(query)
        return [
            RetrievalResult(
                index=int(document.metadata.get("chunk_index", index)),
                text=document.page_content,
                score=None,  # Retriever.invoke returns Documents, not scores.
                method="semantic",
            )
            for index, document in enumerate(documents[:limit])
        ]

    def build_context(self, query: str, *, limit: int = 6) -> str:
        """Join retrieved documents into the context passed to Gemini."""

        results = self.retrieve(query, limit=limit)
        return "\n\n".join(
            f"[Resume chunk {result.index + 1}]\n{result.text}"
            for result in results
        )

    def answer(self, request: AIRequest) -> str:
        """Stages 6-8: build prompt -> call Gemini -> parse string output."""

        if not self.config.ai_ready:
            raise ValueError("GEMINI_API_KEY is required for AI coaching.")

        # This matches prompt.pipe(llm).pipe(new StringOutputParser()) in JS.
        prompt = build_career_prompt()
        llm = ChatGoogleGenerativeAI(
            api_key=self.config.gemini_api_key,
            model=self.config.gemini_model,
            max_output_tokens=2_500,
            max_retries=2,
        )
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke(request.prompt_values())
        if not result.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return result.strip()
