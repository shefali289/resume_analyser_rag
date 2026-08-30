"""CareerLens AI — a focused Streamlit interface for resume RAG and GenAI."""

from __future__ import annotations

from hashlib import sha256
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from resume_analyser.ai import AIRequest
from resume_analyser.ats import ATSAnalysis, analyze_resume
from resume_analyser.config import AppConfig
from resume_analyser.parsers import (
    ParsedResume,
    ResumeParseError,
    parse_resume,
    redact_personal_data,
)
from resume_analyser.rag import ResumeRAGPipeline, count_stored_chunks
from resume_analyser.reports import build_markdown_report


APP_TITLE = "CareerLens AI"
INSIGHT_QUERY = "career strengths achievements role fit technical skills gaps impact"
QUICK_QUESTIONS = (
    "What are the strongest parts of this resume?",
    "Which roles best match this experience?",
    "What important evidence is missing?",
)


def _inject_styles() -> None:
    """Apply a polished visual system without adding UI complexity."""

    st.markdown(
        """
        <style>
        :root {
            --ink: #f8fafc;
            --muted: #9aa7bd;
            --violet: #8b5cf6;
            --cyan: #22d3ee;
            --emerald: #34d399;
            --panel: rgba(13, 21, 38, .78);
        }
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(124, 58, 237, .24), transparent 35rem),
                radial-gradient(circle at 100% 20%, rgba(6, 182, 212, .16), transparent 32rem),
                linear-gradient(180deg, #060a14 0%, #09101f 100%);
        }
        .block-container { max-width: 1160px; padding-top: 2rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 13, 26, .98), rgba(12, 20, 36, .98));
            border-right: 1px solid rgba(148, 163, 184, .12);
        }
        .hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(139, 92, 246, .38);
            border-radius: 28px;
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.35rem;
            background: linear-gradient(130deg, rgba(76, 29, 149, .62), rgba(8, 145, 178, .28));
            box-shadow: 0 28px 90px rgba(0, 0, 0, .34);
        }
        .hero::after {
            content: ""; position: absolute; width: 240px; height: 240px;
            right: -75px; top: -105px; border-radius: 999px;
            background: rgba(34, 211, 238, .16); filter: blur(3px);
        }
        .eyebrow { color: #67e8f9; letter-spacing: .15em; font-size: .73rem; font-weight: 850; }
        .hero h1 { color: white; margin: .35rem 0 .45rem; font-size: clamp(2.4rem, 5vw, 4.2rem); }
        .hero p { color: #dbeafe; font-size: 1.06rem; max-width: 750px; margin: 0; }
        .feature-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1.2rem 0; }
        .feature-card, .pipeline-banner {
            background: var(--panel); border: 1px solid rgba(148, 163, 184, .16);
            border-radius: 20px; padding: 1.25rem;
            box-shadow: inset 0 1px rgba(255,255,255,.035), 0 14px 40px rgba(0,0,0,.14);
        }
        .feature-card:first-child { border-color: rgba(34, 211, 238, .28); }
        .feature-card:last-child { border-color: rgba(139, 92, 246, .34); }
        .feature-card h3 { color: white; margin: .3rem 0 .45rem; font-size: 1.06rem; }
        .feature-card p { color: var(--muted); margin: 0; line-height: 1.55; }
        .feature-icon { font-size: 1.55rem; }
        .pipeline-banner { color: #cbd5e1; text-align: center; font-size: .9rem; }
        .pipeline-banner span { color: #67e8f9; font-weight: 750; }
        .status-dot { display:inline-block; width:.58rem; height:.58rem; border-radius:50%; margin-right:.45rem; }
        .status-on { background: var(--emerald); box-shadow: 0 0 12px rgba(52,211,153,.75); }
        .status-off { background: #64748b; }
        .privacy-note { color:#cbd5e1; font-size:.82rem; line-height:1.48; }
        .resume-grid {
            display:grid; grid-template-columns:minmax(280px, 2fr) 1fr 1fr;
            gap:1rem; margin:1rem 0 1.2rem;
        }
        .meta-card, .tech-card {
            background:rgba(13,21,38,.76); border:1px solid rgba(148,163,184,.14);
            border-radius:17px; padding:1rem 1.1rem; min-width:0;
        }
        .meta-label, .tech-label {
            color:#a5b4cf; font-size:.75rem; font-weight:750;
            letter-spacing:.06em; text-transform:uppercase;
        }
        .meta-value { color:white; font-size:1.6rem; margin-top:.35rem; line-height:1.2; }
        .meta-value.filename {
            font-size:1.03rem; line-height:1.42; overflow-wrap:anywhere; word-break:break-word;
        }
        .tech-grid {
            display:grid; grid-template-columns:repeat(5, minmax(0, 1fr));
            gap:.7rem; margin:.7rem 0 1rem;
        }
        .tech-card { padding:.85rem; border-color:rgba(34,211,238,.17); }
        .tech-value {
            color:#e0f2fe; font-size:.84rem; font-weight:700; margin-top:.32rem;
            line-height:1.35; overflow-wrap:anywhere;
        }
        .chip {
            display:inline-block; border-radius:999px; padding:.32rem .68rem; margin:.18rem;
            font-size:.78rem; font-weight:650; border:1px solid rgba(148,163,184,.22);
        }
        .chip-good { color:#a7f3d0; background:rgba(16,185,129,.12); }
        .chip-gap { color:#fecaca; background:rgba(239,68,68,.12); }
        div[data-testid="stMetric"] {
            background:rgba(13,21,38,.76); border:1px solid rgba(148,163,184,.14);
            padding:.75rem 1rem; border-radius:17px;
        }
        div[data-testid="stTabs"] button { font-weight: 720; }
        div.stButton > button[kind="primary"] {
            border: 0; background: linear-gradient(90deg, #7c3aed, #0891b2);
            box-shadow: 0 10px 30px rgba(124,58,237,.22);
        }
        @media (max-width: 900px) {
            .feature-grid, .resume-grid { grid-template-columns: 1fr; }
            .tech-grid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_config() -> AppConfig:
    load_dotenv()
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    return AppConfig.from_sources(secrets)


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "resume": None,
        "resume_signature": "",
        "rag_pipeline": None,
        "rag_signature": "",
        "retrieval_mode": "Not indexed",
        "stored_chunks": None,
        "stored_chunks_resume": "",
        "index_error": "",
        "last_context": "",
        "ai_outputs": {},
        "chat_messages": [],
        "upload_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_resume_state() -> None:
    st.session_state.rag_pipeline = None
    st.session_state.rag_signature = ""
    st.session_state.retrieval_mode = "Not indexed"
    st.session_state.stored_chunks = None
    st.session_state.stored_chunks_resume = ""
    st.session_state.index_error = ""
    st.session_state.last_context = ""
    st.session_state.ai_outputs = {}
    st.session_state.chat_messages = []


def _status(label: str, ready: bool) -> None:
    class_name = "status-on" if ready else "status-off"
    state = "Ready" if ready else "Not configured"
    st.markdown(
        f'<div class="privacy-note"><span class="status-dot {class_name}"></span>'
        f"{escape(label)} · {state}</div>",
        unsafe_allow_html=True,
    )


def _render_landing(engineer_mode: bool) -> None:
    if not engineer_mode:
        st.markdown(
            "<style>.pipeline-banner { display: none; }</style>",
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="feature-grid">
          <div class="feature-card">
            <div class="feature-icon">💬</div>
            <h3>RAG Resume Chat</h3>
            <p>Ask questions about the resume. Every answer is grounded in relevant chunks retrieved from Supabase pgvector.</p>
          </div>
          <div class="feature-card">
            <div class="feature-icon">✨</div>
            <h3>GenAI Career Insights</h3>
            <p>Generate one focused analysis covering strengths, suitable roles, evidence gaps, and high-impact improvements.</p>
          </div>
        </div>
        <div class="pipeline-banner">
          Resume → <span>Chunks</span> → Hugging Face embeddings → <span>Supabase vectors</span> → Retrieval → Gemini
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Upload a TXT, PDF, or DOCX resume from the sidebar. A job description is optional.")


def _render_resume_summary(resume: ParsedResume) -> None:
    """Show resume metadata without truncating long filenames."""

    st.markdown(
        f"""
        <div class="resume-grid">
          <div class="meta-card">
            <div class="meta-label">Resume</div>
            <div class="meta-value filename">{escape(resume.filename)}</div>
          </div>
          <div class="meta-card">
            <div class="meta-label">Words</div>
            <div class="meta-value">{resume.word_count:,}</div>
          </div>
          <div class="meta-card">
            <div class="meta-label">Pages</div>
            <div class="meta-value">{resume.page_count}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _refresh_stored_chunks(resume: ParsedResume, config: AppConfig) -> None:
    """Ask Supabase how many chunks are really stored for this resume."""

    st.session_state.stored_chunks_resume = resume.resume_id
    try:
        st.session_state.stored_chunks = count_stored_chunks(config, resume.resume_id)
        st.session_state.index_error = ""
    except Exception as exception:
        st.session_state.stored_chunks = None
        st.session_state.index_error = f"Could not read the Supabase index: {exception}"


def _stored_chunks(resume: ParsedResume, config: AppConfig) -> int | None:
    """Return the verified chunk count, querying Supabase once per resume.

    ``None`` means the count is unknown: Supabase is unconfigured or the
    lookup failed, so the caller falls back to this session's own state.
    """

    if not config.supabase_ready:
        return None
    if st.session_state.stored_chunks_resume != resume.resume_id:
        _refresh_stored_chunks(resume, config)
    return st.session_state.stored_chunks


def _render_index_controls(
    resume: ParsedResume,
    config: AppConfig,
    allow_external_ai: bool,
    redact_pii: bool,
    indexed: bool,
) -> None:
    """Let engineers index on demand instead of waiting for the first question."""

    # Indexing sends resume text to Hugging Face and Supabase, so it needs the
    # same consent toggle that gates Chat and Insights.
    can_index = allow_external_ai and config.vector_db_ready

    index_column, refresh_column = st.columns(2)
    if index_column.button(
        "Re-index resume" if indexed else "Index resume",
        use_container_width=True,
        disabled=not can_index,
        help="Chunk the resume, embed it, and upsert the vectors into Supabase.",
    ):
        try:
            with st.spinner("Embedding chunks and writing vectors to Supabase…"):
                _ensure_pipeline(resume, config, redact_pii, force=True)
                _refresh_stored_chunks(resume, config)
        except Exception as exception:
            st.session_state.index_error = f"Indexing failed: {exception}"
        st.rerun()

    if refresh_column.button(
        "Refresh status",
        use_container_width=True,
        disabled=not config.supabase_ready,
        help="Re-count the rows Supabase holds for this resume.",
    ):
        _refresh_stored_chunks(resume, config)
        st.rerun()

    if st.session_state.index_error:
        st.error(st.session_state.index_error)
    if not config.vector_db_ready:
        st.caption(
            "Indexing needs: " + ", ".join(config.missing_vector_variables),
        )
    elif not allow_external_ai:
        st.caption("Enable RAG + GenAI in the sidebar to index this resume.")


def _render_engineer_panel(
    config: AppConfig,
    resume: ParsedResume | None = None,
    allow_external_ai: bool = False,
    redact_pii: bool = True,
) -> None:
    """Expose the concrete RAG implementation only in Engineer mode."""

    if resume is None:
        index_status, chunk_count = "No resume loaded", 0
    else:
        stored = _stored_chunks(resume, config)
        if stored is None:
            # Supabase could not answer, so report what this session did itself.
            pipeline: ResumeRAGPipeline | None = st.session_state.rag_pipeline
            chunk_count = len(pipeline.documents) if pipeline else 0
            index_status = st.session_state.retrieval_mode if pipeline else "Unknown"
        else:
            chunk_count = stored
            index_status = "Supabase semantic" if stored else "Not indexed"
    st.markdown("### Engineer pipeline")
    st.markdown(
        f"""
        <div class="tech-grid">
          <div class="tech-card">
            <div class="tech-label">Chunking</div>
            <div class="tech-value">500 chars<br>50 overlap</div>
          </div>
          <div class="tech-card">
            <div class="tech-label">Embeddings</div>
            <div class="tech-value">{escape(config.hf_embedding_model)}<br>384 dimensions</div>
          </div>
          <div class="tech-card">
            <div class="tech-label">Vector database</div>
            <div class="tech-value">Supabase pgvector<br>{escape(config.supabase_table)}</div>
          </div>
          <div class="tech-card">
            <div class="tech-label">Retriever</div>
            <div class="tech-value">{escape(config.supabase_query)}<br>top 6 chunks</div>
          </div>
          <div class="tech-card">
            <div class="tech-label">Current index</div>
            <div class="tech-value">{escape(index_status)}<br>{chunk_count} chunks</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Generator: {config.gemini_model} · Stable document IDs make Supabase upserts idempotent.",
    )
    if resume is not None:
        _render_index_controls(
            resume,
            config,
            allow_external_ai,
            redact_pii,
            bool(chunk_count),
        )


def _chips(items: list[str], kind: str) -> None:
    if not items:
        st.caption("None detected")
        return
    st.markdown(
        "".join(
            f'<span class="chip chip-{kind}">{escape(item)}</span>'
            for item in items
        ),
        unsafe_allow_html=True,
    )


def _ensure_pipeline(
    resume: ParsedResume,
    config: AppConfig,
    redact_pii: bool,
    force: bool = False,
) -> ResumeRAGPipeline:
    """Index the resume once and reuse its Supabase retriever in this session.

    ``force`` re-runs the embedding and upsert even when this session already
    holds a matching pipeline, which is what the explicit re-index button needs.
    """

    signature = f"{resume.resume_id}:{redact_pii}:{config.hf_embedding_model}"
    if (
        not force
        and st.session_state.rag_pipeline is not None
        and st.session_state.rag_signature == signature
    ):
        return st.session_state.rag_pipeline

    external_text = redact_personal_data(resume.text) if redact_pii else resume.text
    pipeline = ResumeRAGPipeline(config)
    pipeline.prepare_resume(
        external_text,
        resume_id=resume.resume_id,
        filename=resume.filename,
    )
    st.session_state.rag_pipeline = pipeline
    st.session_state.rag_signature = signature
    st.session_state.retrieval_mode = pipeline.mode
    # Chat and Insights also index, so keep the engineer status card in step.
    st.session_state.stored_chunks = len(pipeline.documents)
    st.session_state.stored_chunks_resume = resume.resume_id
    return pipeline


def _configuration_error(
    config: AppConfig,
    allow_external_ai: bool,
) -> str | None:
    if not allow_external_ai:
        return "Enable RAG + GenAI in the sidebar first."
    if not config.deployment_ready:
        return "Missing configuration: " + ", ".join(
            config.missing_deployment_variables,
        )
    return None


def _render_chat(
    resume: ParsedResume,
    config: AppConfig,
    job_description: str,
    target_role: str,
    allow_external_ai: bool,
    redact_pii: bool,
    engineer_mode: bool,
) -> None:
    st.subheader("Chat with your resume")
    st.caption(
        "Supabase retrieves the most relevant resume chunks before Gemini answers. "
        "The optional role and job description are used only as extra context.",
    )

    quick_columns = st.columns(3)
    quick_question = ""
    for index, suggestion in enumerate(QUICK_QUESTIONS):
        if quick_columns[index].button(
            suggestion,
            key=f"quick_{index}",
            use_container_width=True,
        ):
            quick_question = suggestion

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything grounded in this resume…") or quick_question
    if not question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    error = _configuration_error(config, allow_external_ai)
    if error:
        answer = error
    else:
        try:
            with st.spinner("Retrieving resume evidence from Supabase…"):
                pipeline = _ensure_pipeline(resume, config, redact_pii)
                context = pipeline.build_context(question, limit=6)
                st.session_state.last_context = context
                answer = pipeline.answer(
                    AIRequest(
                        task="chat",
                        question=question,
                        context=context,
                        job_description=job_description,
                        target_role=target_role,
                    ),
                )
        except Exception as exception:
            answer = f"The RAG assistant could not answer: {exception}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    if engineer_mode and st.session_state.last_context:
        with st.expander("View retrieved evidence"):
            st.code(st.session_state.last_context, language="text")


def _render_genai(
    resume: ParsedResume,
    config: AppConfig,
    analysis: ATSAnalysis | None,
    job_description: str,
    target_role: str,
    allow_external_ai: bool,
    redact_pii: bool,
) -> None:
    st.subheader("Generate career insights")
    st.caption(
        "One grounded report: strengths, role fit, evidence gaps, and the most "
        "valuable resume improvements.",
    )

    left, right = st.columns([1.35, .65])
    with left:
        st.markdown(
            """
            **The report will cover**

            - strongest evidence and differentiators;
            - suitable roles and positioning;
            - unsupported or weak claims;
            - five actionable improvements.
            """,
        )
    with right:
        generate = st.button(
            "✨ Generate insights",
            type="primary",
            use_container_width=True,
        )

    if generate:
        error = _configuration_error(config, allow_external_ai)
        if error:
            st.warning(error)
        else:
            try:
                with st.spinner("Retrieving evidence and generating your report…"):
                    pipeline = _ensure_pipeline(resume, config, redact_pii)
                    context = pipeline.build_context(INSIGHT_QUERY, limit=6)
                    st.session_state.last_context = context
                    st.session_state.ai_outputs["analysis"] = pipeline.answer(
                        AIRequest(
                            task="analysis",
                            context=context,
                            job_description=job_description,
                            target_role=target_role,
                        ),
                    )
            except Exception as exception:
                st.error(f"Insight generation failed: {exception}")

    output = st.session_state.ai_outputs.get("analysis")
    if output:
        st.markdown(output)
        report = build_markdown_report(
            resume,
            analysis,
            target_role,
            st.session_state.ai_outputs,
        )
        st.download_button(
            "Download insights",
            data=report,
            file_name=f"{Path(resume.filename).stem}-career-insights.md",
            mime="text/markdown",
            use_container_width=True,
        )


def _render_transparency(
    resume: ParsedResume,
    analysis: ATSAnalysis | None,
    config: AppConfig,
    has_job_description: bool,
    engineer_mode: bool,
) -> None:
    st.markdown("### Transparent by design")
    if engineer_mode:
        ats_expander, vectors_expander = st.columns(2)
    else:
        ats_expander = st.container()
        vectors_expander = None

    with ats_expander:
        with st.expander("How ATS scoring works", expanded=False):
            st.markdown(
                """
                The score is a deterministic heuristic—not an employer ATS result:

                - **50 points:** job-description keyword coverage
                - **15 points:** standard resume sections
                - **15 points:** action-led and quantified bullets
                - **10 points:** readability and sensible length
                - **10 points:** email, phone, and LinkedIn checks
                """,
            )
            if not has_job_description:
                st.info("Add an optional job description to calculate the ATS match.")
            elif analysis:
                st.metric("Estimated ATS match", f"{analysis.score}/100")
                for component in analysis.components:
                    st.caption(
                        f"{component.label}: {component.score:.1f}/{component.maximum:.0f}",
                    )
                    st.progress(component.score / component.maximum)
                st.markdown("**Matched terms**")
                _chips(analysis.matched_keywords, "good")
                st.markdown("**Potential gaps**")
                _chips(analysis.missing_keywords, "gap")

    if vectors_expander is not None:
        with vectors_expander, st.expander(
            "How vectors are stored and checked",
            expanded=False,
        ):
            stored = _stored_chunks(resume, config)
            if stored is None:
                pipeline: ResumeRAGPipeline | None = st.session_state.rag_pipeline
                chunk_count = len(pipeline.documents) if pipeline else 0
            else:
                chunk_count = stored
            st.markdown(
                f"""
                - **Store:** Supabase pgvector
                - **Table:** `{config.supabase_table}`
                - **Search RPC:** `{config.supabase_query}`
                - **Embedding model:** `{config.hf_embedding_model}`
                - **Dimensions:** `384`
                - **Resume ID:** `{resume.resume_id[:12]}…`
                - **Current indexed chunks:** `{chunk_count or 'not indexed yet'}`

                Open **Supabase → Table Editor → documents** to inspect rows, or run:
                """,
            )
            st.code(
                """select
  id,
  metadata->>'filename' as filename,
  metadata->>'resume_id' as resume_id,
  metadata->>'chunk_index' as chunk_index,
  left(content, 100) as preview,
  vector_dims(embedding) as dimensions
from documents
order by id desc
limit 20;""",
                language="sql",
            )
            st.caption(
                "The embedding column is a numeric vector, so inspect dimensions and "
                "content metadata instead of expecting readable vector values.",
            )


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} · Resume Intelligence",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()
    _init_state()
    config = _load_config()

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">PERSISTENT RAG · GENERATIVE CAREER INTELLIGENCE</div>
          <h1>CareerLens AI</h1>
          <p>Chat with your resume or generate a focused career report. Every AI response is grounded in evidence retrieved from Supabase.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Resume workspace")
        view_mode = st.radio(
            "Interface mode",
            ("General", "Engineer"),
            horizontal=True,
            help=(
                "General keeps the interface resume-focused. Engineer reveals "
                "the RAG pipeline, retrieval evidence, and service configuration."
            ),
        )
        engineer_mode = view_mode == "Engineer"
        st.caption(
            "Resume insights only"
            if not engineer_mode
            else "RAG implementation and diagnostics enabled"
        )

        uploaded_file = st.file_uploader(
            "Upload resume",
            type=["txt", "pdf", "docx"],
            help="Maximum 8 MB. The file is parsed in memory.",
            key=f"resume_upload_{st.session_state.upload_version}",
        )
        if st.button("Clear local session", use_container_width=True):
            st.session_state.resume = None
            st.session_state.resume_signature = ""
            st.session_state.upload_version += 1
            _reset_resume_state()
            st.rerun()

        st.divider()
        st.subheader("RAG controls")
        allow_external_ai = st.toggle(
            "Enable RAG + GenAI",
            value=False,
            disabled=not config.deployment_ready,
            help="Required to embed, store, retrieve, and generate responses.",
        )
        redact_pii = st.toggle("Redact contact details", value=True)
        privacy_message = (
            "When enabled, resume chunks are embedded by Hugging Face, persisted "
            "in Supabase, and relevant evidence is sent to Gemini."
            if engineer_mode
            else "When enabled, the assistant securely processes the resume to "
            "answer questions and generate career insights."
        )
        st.markdown(
            f'<p class="privacy-note">{privacy_message}</p>',
            unsafe_allow_html=True,
        )

        if engineer_mode:
            st.divider()
            st.subheader("Services")
            _status("Gemini", config.ai_ready)
            _status("Hugging Face", config.semantic_search_ready)
            _status("Supabase", config.supabase_ready)
        if not config.deployment_ready:
            st.error(
                "Missing configuration: "
                + ", ".join(config.missing_deployment_variables),
            )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        signature = sha256(file_bytes).hexdigest()
        if signature != st.session_state.resume_signature:
            try:
                st.session_state.resume = parse_resume(file_bytes, uploaded_file.name)
                st.session_state.resume_signature = signature
                _reset_resume_state()
            except ResumeParseError as exception:
                st.error(str(exception))
                st.session_state.resume = None

    resume: ParsedResume | None = st.session_state.resume
    if resume is None:
        _render_landing(engineer_mode)
        if engineer_mode:
            _render_engineer_panel(config)
        return

    st.success(
        f"Loaded {resume.filename} · {resume.file_type} · "
        f"{resume.word_count:,} words",
    )
    _render_resume_summary(resume)
    if engineer_mode:
        _render_engineer_panel(config, resume, allow_external_ai, redact_pii)

    with st.expander("Optional tailoring context", expanded=False):
        st.caption(
            "Both fields are optional. Leave them blank for resume-only chat and insights.",
        )
        target_column, job_column = st.columns([.75, 1.5])
        with target_column:
            target_role = st.text_input(
                "Target role (optional)",
                placeholder="e.g. AI Engineer",
            )
        with job_column:
            job_description = st.text_area(
                "Job description (optional)",
                height=145,
                placeholder="Paste a job description only when you want tailored matching…",
            )

    has_job_description = bool(job_description.strip())
    analysis = (
        analyze_resume(resume.text, job_description)
        if has_job_description
        else None
    )

    rag_tab, genai_tab = st.tabs(["💬 RAG Resume Chat", "✨ GenAI Insights"])
    with rag_tab:
        _render_chat(
            resume,
            config,
            job_description,
            target_role,
            allow_external_ai,
            redact_pii,
            engineer_mode,
        )
    with genai_tab:
        _render_genai(
            resume,
            config,
            analysis,
            job_description,
            target_role,
            allow_external_ai,
            redact_pii,
        )

    _render_transparency(
        resume,
        analysis,
        config,
        has_job_description,
        engineer_mode,
    )


if __name__ == "__main__":
    main()
