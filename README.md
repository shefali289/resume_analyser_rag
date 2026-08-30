# CareerLens AI — Resume Analyser RAG

CareerLens is a portfolio-grade resume intelligence app combining explainable
ATS analysis with persistent retrieval-augmented generation. Upload a TXT, PDF,
or DOCX resume, compare it with a target job, and generate grounded coaching
with LangChain, Hugging Face, Supabase pgvector, and Google Gemini.

![CareerLens RAG architecture](docs/architecture.svg)

## Features

- Explainable ATS score with keyword, structure, readability, contact, and
  quantified-impact checks.
- Persistent RAG: every AI workflow uses Supabase pgvector. There is no
  in-memory vector-store or lexical-retrieval path.
- Stable document IDs make ingestion idempotent across Vercel/serverless
  restarts.
- Two focused AI workflows: grounded resume chat and a concise career-insights
  report covering strengths, role fit, evidence gaps, and improvements.
- Two interface modes: General keeps the experience resume-focused, while
  Engineer exposes chunking, embeddings, Supabase pgvector, retrieval settings,
  indexed chunk counts, service status, and retrieved evidence.
- Engineer mode reports the indexed chunk count read back from Supabase, and
  offers explicit **Index resume** and **Refresh status** controls so indexing
  never has to wait for the first question.
- Optional contact-detail redaction before resume text leaves the app.
- Downloadable Markdown reports.
- Streamlit UI packaged as a Vercel container Function.

## RAG implementation

The commented Python implementation in `resume_analyser/rag.py` intentionally
matches the original `index.js` prototype:

1. `RecursiveCharacterTextSplitter` creates 500-character chunks with a
   50-character overlap.
2. `HuggingFaceEndpointEmbeddings` uses
   `sentence-transformers/all-MiniLM-L6-v2` to produce 384-dimensional vectors.
3. `SupabaseVectorStore.from_documents` upserts the chunks into `documents`.
4. `vector_store.as_retriever()` creates a resume-scoped retriever.
5. `retriever.invoke(query)` selects the relevant evidence.
6. `PromptTemplate | ChatGoogleGenerativeAI | StrOutputParser` produces the
   grounded result.

ATS scoring remains local. When external processing is enabled, the RAG path
always stores vectors in Supabase before retrieval and Gemini generation.

### When indexing happens

Uploading a resume only parses it in memory; no text leaves the app at that
point. Chunking, embedding, and the Supabase upsert run when either:

- **Index resume** is pressed in the Engineer pipeline panel; or
- the first Resume Chat question or GenAI Insights report is requested, which
  indexes on demand before retrieving.

Both paths require the **Enable RAG + GenAI** toggle, because both send resume
text to Hugging Face and Supabase. The button is disabled until that toggle is
on and the vector-store variables are configured.

The **Current index** card is not session state. It is a metadata-only
`count(*)` of the Supabase rows whose `metadata->>'resume_id'` matches the
uploaded resume, so a freshly restarted session still reports a resume that was
indexed earlier. The count is fetched once per resume; **Refresh status**
re-reads it, and neither call downloads embeddings or resume text.

## Local setup

Python 3.12 is recommended because it matches the deployment container.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.sample .env
```

Add the required credentials to `.env`, run the Supabase schema described
below, and start the application:

```powershell
python scripts/check_deployment.py --live
streamlit run streamlit_app.py
```

If PowerShell blocks activation:

```powershell
.\.venv\Scripts\python.exe scripts\check_deployment.py --live
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Required environment variables

All production values must be configured in Vercel Project Settings. Do not
commit `.env` or `.streamlit/secrets.toml`.

| Variable | Required | Value |
| --- | --- | --- |
| `HF_API_KEY` | Yes | Hugging Face token with Inference Providers access |
| `HF_EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` |
| `SUPABASE_URL` | Yes | `https://PROJECT_REF.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Server-only secret/service-role key |
| `SUPABASE_TABLE` | No | `documents` |
| `SUPABASE_QUERY` | No | `match_documents` |
| `GEMINI_API_KEY` | Yes | Google AI Studio API key |
| `GEMINI_MODEL` | No | `gemini-3.7-flash` |

`HUGGINGFACEHUB_API_TOKEN`, `GOOGLE_API_KEY`, and the older `SUPABASE_KEY` are
accepted as compatibility aliases. Prefer the canonical names in the table.

The Supabase URL must be the project base URL without `/rest/v1`. Never use a
browser-visible/public variable such as `NEXT_PUBLIC_SUPABASE_SERVICE_ROLE_KEY`.
The service-role key stays inside the server-side Streamlit container.

## Supabase setup

1. Create a Supabase project.
2. Open its SQL editor.
3. Run the complete [`supabase/schema.sql`](supabase/schema.sql) script.
4. Copy the project URL and server-side service-role/secret key into `.env` and
   Vercel Project Settings.
5. Run `python scripts/check_deployment.py --live`.

The schema enables pgvector, creates the `documents` table, adds a GIN metadata
index, enables row-level security, and creates `match_documents`. The embedding
column has 384 dimensions and must match the configured Hugging Face model.

The table intentionally has no public RLS policy. Access is through the
server-side key only.

## Deploy to Vercel

Vercel now supports OCI container Functions and detects `Dockerfile.vercel` at
the repository root. The container runs Streamlit on Vercel's injected `PORT`.

1. Push this repository to GitHub.
2. Import the repository into Vercel.
3. Ensure Fluid Compute is enabled for the project.
4. In **Settings → Environment Variables**, add every required variable from
   the table above. Select Production and Preview only where the credentials
   should be available.
5. Use a separate Supabase project/key for untrusted preview deployments when
   possible.
6. Deploy. Vercel automatically builds `Dockerfile.vercel`; no build or output
   directory override is required.
7. Open the deployment, confirm all three service indicators show **Ready**,
   upload a non-sensitive test resume, and run one Resume Chat question.

Vercel container Functions are stateless and can scale down. Persistent vectors
remain in Supabase, while Streamlit UI session state may reset when a Function
instance is replaced. Stable chunk IDs prevent duplicate rows after a reconnect
or cold start.

### Test the production container locally

If Docker is installed:

```powershell
docker build -f Dockerfile.vercel -t careerlens-ai .
docker run --rm -p 8501:80 --env-file .env careerlens-ai
```

Then open `http://localhost:8501`.

## Safe deployment preflight

Static validation checks required keys and URL structure without printing any
secret:

```powershell
python scripts/check_deployment.py
```

Live validation sends only harmless test text—not a resume—to Hugging Face and
Gemini and performs a read-only Supabase similarity-search RPC:

```powershell
python scripts/check_deployment.py --live
```

The preflight verifies:

- required variables are present;
- Hugging Face returns 384-dimensional vectors;
- the Supabase URL, key, table, and RPC work together;
- the configured Gemini model accepts a request.

## Tests

The automated suite makes no network calls; Hugging Face, Supabase, and Gemini
are all faked at their client boundaries.

```powershell
python -m unittest discover -s tests -v
python -m pip check
node --check index.js
```

19 tests cover ATS scoring, configuration and its aliases, resume parsing and
redaction, the split/embed/retrieve/generate pipeline, the Supabase chunk
count, and Streamlit rendering. The `tests/test_streamlit_smoke.py` cases drive
the real app through `streamlit.testing.v1.AppTest`, asserting that the
Engineer panel reports the stored chunk count, exposes the index controls,
blocks indexing until external processing is enabled, and surfaces indexing
failures instead of raising.

## Project structure

```text
resume_analyser_rag/
|-- .streamlit/
|   |-- config.toml
|   `-- secrets.toml.example
|-- docs/
|   `-- architecture.svg
|-- resume_analyser/
|   |-- ai.py
|   |-- ats.py
|   |-- config.py
|   |-- parsers.py
|   |-- rag.py
|   `-- reports.py
|-- scripts/
|   `-- check_deployment.py
|-- supabase/
|   `-- schema.sql
|-- tests/
|   |-- test_ats.py
|   |-- test_config.py
|   |-- test_parsers.py
|   |-- test_rag.py
|   `-- test_streamlit_smoke.py
|-- .dockerignore
|-- .env.sample
|-- Dockerfile.vercel
|-- index.js
|-- requirements.txt
`-- streamlit_app.py
```

## Legacy Node.js prototype

The original CLI remains available:

```powershell
npm.cmd install
node index.js
```

It uses the same environment variables, embedding model, Supabase table/RPC,
and Gemini model as the Streamlit app.

## Disclaimer

ATS scores are explainable estimates, not guarantees of employer-system
ranking. Review all generated content and add keywords or metrics only when they
truthfully describe the candidate's experience.
