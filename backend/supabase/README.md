# Supabase setup

Supabase is optional. In local mode (the default after `./bonclay setup`) everything is stored on your computer in
`backend/data` and none of this is needed. Use Supabase for hosted mode, where several users share one deployment.

## Migrations

Run the files in `migrations/` **in order, 000 to 011**, on a fresh Supabase project. Nothing else needs to be written by
hand: the `match_document_chunks` RPC is created by `000`. Either paste each file into the
Supabase SQL editor, or from `backend/` run `supabase db push` (Supabase CLI, linked to your project).

| File | What it does |
| --- | --- |
| `000_core_schema.sql` | `vector` extension, all core tables (`projects`, `voices`, `agents`, `documents`, `document_chunks` with `vector(384)` + HNSW index, `conversations`, `conversation_turns`, `evaluation_runs`, `generations`, `api_keys`), the `match_document_chunks` RPC, and the private `audio-assets` storage bucket |
| `001_rls.sql` | Owner-only RLS for `generations`, `embeddings`, `api_keys` |
| `002_agents_user_scope.sql` | Per-user ownership + RLS for agents, conversations, documents and their child tables |
| `003_projects_voices_rls.sql` | RLS for `projects` and `voices` |
| `004_consents.sql` | `consents` table and `voices.consent_id` |
| `005_turn_metrics.sql` | Per-turn timings, engine and model on `conversation_turns` (the reply-time panel reads these) |
| `006_consent_speaker.sql` | `consents.speaker_score` and `consents.challenge_id` (the one-time sentence and the speaker match) |
| `007_agent_disclose_ai.sql` | `agents.disclose_ai`: agents say they are an AI on the first reply (on by default) |
| `008_receipts.sql` | `receipts`: one signed provenance receipt per generation and agent reply, read by the public `/verify` page through the backend |
| `009_agent_tools.sql` | `agents.tools` (MCP tool servers an agent may use) and `conversation_turns.tool_calls` |
| `011_embeddings_1024.sql` | `document_chunks.embedding_1024 vector(1024)` + `embed_model`, an HNSW index on it and the `match_document_chunks_1024` RPC, for the BAAI/bge-m3 embeddings. The 384-d column and RPC stay as they are |
| `020_match_question_rows.sql` | Both match RPCs also return `kind`, `parent_chunk_id` and `header` (columns from `018`, which must run first) plus the parent passage of every question row they return: the backend scores question rows but hands the model the parent passage. The backend also works with the older RPCs (every row is then a passage) |

All files are idempotent, so re-running them is safe. `000` must run first because later files ALTER its tables.

### Embeddings (knowledge search)

The backend embeds knowledge with `EMBEDDING_MODEL`, `BAAI/bge-m3` by default (1024 dimensions, MIT licence, strong on
Hindi, English and mixed text). `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions) stays
selectable. A pgvector column has a fixed size, so each size has its own column and match function, and the backend
picks them from the length of the query vector:

| Dimension | Column on `document_chunks` | RPC | Created by |
| --- | --- | --- | --- |
| 384 (MiniLM) | `embedding` | `match_document_chunks` | `000` |
| 1024 (BGE-M3, default) | `embedding_1024` | `match_document_chunks_1024` | `011` |

**Run `011` before deploying a backend that uses the default model**, otherwise uploads and searches in hosted mode fail
(the search then answers with no knowledge). Chunks uploaded earlier keep their 384-d vectors; on the first search of an
agent after the switch the backend embeds that agent's chunks into `embedding_1024` (once per process, bounded batches),
so no manual backfill is needed. To go back, set `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: the
384-d data was never touched. Any other model needs a new column and RPC of its size (add its dimension to
`CLOUD_VECTOR_TARGETS` in `backend.py`).

Search is hybrid: the match function returns a dense shortlist (30 chunks) which is re-ordered by fusing it with a BM25
ranking of the same chunks (weighted score fusion; `RETRIEVAL_FUSION=rrf` for the previous Reciprocal Rank Fusion, k=60). In local mode BM25 runs over every chunk of the agent.
`HYBRID_SEARCH=off` turns the BM25 part off; `RERANKER=on` adds the `BAAI/bge-reranker-v2-m3` cross-encoder over the top 20
(off by default: it adds about a second per query on this machine's CPU).

The `audio-assets` bucket is private. `backend.py` gives clients signed, time-limited playback links
(`create_signed_url`, valid for `SIGNED_URL_EXPIRES_S` seconds, default 3600). The generations table stores the storage
path (`storage:<path>`) and signs it each time the history is read, so old entries keep playing.

## Backend environment variables

Copy `backend/.env.example` to `backend/.env` (the backend also reads `../frontend/.env.local`). The three Supabase variables
are required only in hosted mode; local mode runs without them.

| Variable | Needed | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_SUPABASE_URL` | hosted mode | Project URL, e.g. `https://your-project.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | hosted mode, secret | Backend DB and storage access (bypasses RLS, so never expose it) |
| `SUPABASE_JWT_SECRET` | hosted mode, for authenticated routes, secret | Verifies the user JWTs sent to the API |
| `GROQ_API_KEY` | optional | Faster agent generation, chat, evaluation judge and speech recognition. Without it the agents use a local model on a Mac with Apple Silicon, or any OpenAI-compatible server |
| `SIGNED_URL_EXPIRES_S` | optional | Lifetime of signed audio links in seconds (default 3600) |
| `HF_TOKEN` | optional | Downloading gated Hugging Face models (Indic-Speak). Chatterbox needs no token |
| `WHISPER_MODEL` | optional | Local speech recognition size when `GROQ_API_KEY` is unset and the Apple GPU path is not used (default `small`) |
| `MAX_GPU_MEMORY_GB` | optional | GPU memory budget for the model manager (default 12) |
| `PYTORCH_CUDA_ALLOC_CONF` | optional | CUDA allocator tuning |
| `MLX_PYTHON`, `MLX_CLONE_MODEL`, `CLONE_BACKEND` | optional | Voice cloning on Apple Silicon / torch fallback |
| `INDIC_SPEAK_PYTHON`, `INDIC_SPEAK_MODEL`, `INDIC_SPEAK_MAX_CHARS`, `INDIC_SPEAK_TIMEOUT`, `INDIC_SPEAK` | optional | Indic-Speak worker settings |

The frontend needs its own `NEXT_PUBLIC_SUPABASE_URL` and anon key in `frontend/.env.local`.
