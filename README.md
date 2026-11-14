# RAG Backend

A fully typed FastAPI backend for document ingestion, semantic retrieval,
multi-turn conversational RAG, and interview booking.

The project uses PostgreSQL, Qdrant, Redis, and Ollama. It does not use FAISS,
Chroma, a UI, or LangChain's `RetrievalQAChain`.

## Features

- Upload `.pdf` and `.txt` documents.
- Select recursive or semantic chunking per upload.
- Generate local embeddings with Ollama `embeddinggemma`.
- Store metadata and chunks in PostgreSQL and vectors in Qdrant.
- Build custom RAG prompts and generate answers with Ollama `gemma3:1b`.
- Return source document, page, chunk, and similarity information.
- Store bounded, expiring multi-turn conversation history in Redis.
- Extract, validate, and persist interview bookings.
- Reject conflicting interview time slots.

## Architecture

```text
PDF/TXT -> extraction -> recursive/semantic chunking -> Ollama embeddings
        -> PostgreSQL metadata/chunks + Qdrant vectors

Chat -> Redis history -> query embedding -> Qdrant retrieval
     -> custom grounded prompt -> Ollama answer + sources

Booking request -> LLM extraction -> Pydantic validation -> PostgreSQL
```

## Technology stack

- Python 3.12, FastAPI, Uvicorn
- SQLAlchemy async, Alembic, PostgreSQL 17
- Qdrant, Redis 7, Ollama
- `uv` for Python and dependency management
- Docker Compose for infrastructure

## Prerequisites

Install Git, Docker, and `uv`. Use Docker Desktop on Windows/macOS or Docker
Engine with the Compose plugin on Linux. Docker must be running before any
`docker compose` command. On Windows, Docker Desktop must use Linux containers
with WSL 2.

Verify the tools:

```bash
git --version
docker --version
docker compose version
uv --version
```

Install `uv` on Windows PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

On macOS with Homebrew:

```bash
brew install uv
```

Linux/macOS installer alternative:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal if `uv` is not immediately found.

## Fresh installation

### 1. Clone the repository

```bash
git clone https://github.com/dhirajs16/rag-backend.git
cd rag-backend
```

### 2. Install Python and dependencies

```bash
uv python install 3.12
uv sync --locked
```

This creates `.venv` and installs the exact dependencies from `uv.lock`. A
separate global `pip install` is unnecessary. The environment also does not
need to be activated when commands are run through `uv run`.

### 3. Create `.env`

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Git Bash, Linux, or macOS:

```bash
cp .env.example .env
```

The defaults expect FastAPI on the host and infrastructure in Docker:

```env
DATABASE_URL="postgresql+psycopg://postgres:postgres@127.0.0.1:5433/rag_backend"
OLLAMA_URL="http://127.0.0.1:11434"
QDRANT_URL="http://127.0.0.1:6333"
REDIS_URL="redis://127.0.0.1:6379/0"
EMBEDDING_MODEL="embeddinggemma"
CHAT_MODEL="gemma3:1b"
```

Never commit `.env`; it is intentionally ignored by Git.

### 4. Start infrastructure

Start Docker Desktop/Engine, then run:

```bash
docker compose up -d postgres adminer qdrant redis ollama
docker compose ps
```

`-d` runs containers in the background. Expected services and host ports:

| Service | Host address | Purpose |
|---|---|---|
| PostgreSQL | `127.0.0.1:5433` | Documents, chunks, bookings |
| Adminer | `http://localhost:9000` | Database administration |
| Qdrant | `http://localhost:6333` | Vector database |
| Redis | `127.0.0.1:6379` | Conversation memory |
| Ollama | `http://localhost:11434` | Embeddings and generation |

PostgreSQL deliberately uses host port `5433`, avoiding conflicts with a local
PostgreSQL server on `5432`. Every service should show `Up`; PostgreSQL and
Redis should become `healthy` after a few seconds.

Inspect a service if it does not start:

```bash
docker compose logs postgres
docker compose logs qdrant
docker compose logs redis
docker compose logs ollama
```

Use `docker compose logs -f SERVICE` for live logs. `Ctrl+C` stops following
logs without stopping the container.

### 5. Download Ollama models

The Ollama container initially has no models:

```bash
docker compose exec ollama ollama pull embeddinggemma
docker compose exec ollama ollama pull gemma3:1b
docker compose exec ollama ollama list
```

`embeddinggemma` is approximately 622 MB and generates 768-dimensional
vectors. `gemma3:1b` is approximately 806 MB. Downloads may take several
minutes. The final list must contain both model names.

### 6. Apply database migrations

```bash
uv run alembic upgrade head
uv run alembic current
```

Expected revision:

```text
b4172e18eac1 (head)
```

This creates `documents`, `document_chunks`, `interview_bookings`, and
Alembic's migration-tracking table.

### 7. Run automated checks

```bash
uv run pytest -q
uv run alembic check
```

Expected:

```text
4 passed
No new upgrade operations detected.
```

### 8. Start FastAPI

```bash
uv run uvicorn rag_backend.main:app --reload
```

Keep the terminal open. Application addresses:

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- OpenAPI: `http://127.0.0.1:8000/openapi.json`
- Health: `http://127.0.0.1:8000/api/v1/health`

Test health from another terminal:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Expected:

```json
{"status":"ok"}
```

## Test the complete workflow

Swagger is the easiest cross-platform interface, especially for file uploads.
Open `http://127.0.0.1:8000/docs`, expand an endpoint, select **Try it out**,
enter the data, and select **Execute**.

### 1. Upload documents with both strategies

Use `POST /api/v1/documents` with:

- `file`: a non-empty `.pdf` or `.txt`
- `chunking_strategy`: `recursive` or `semantic`

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -H "accept: application/json" \
  -F "file=@/absolute/path/to/document.pdf;type=application/pdf" \
  -F "chunking_strategy=recursive"
```

Success returns HTTP `201` with `"status":"completed"`. Upload a different
file using `semantic`. Identical content returns HTTP `409`, even if renamed,
because duplicate detection uses a SHA-256 checksum.

Verify Qdrant:

```bash
curl http://127.0.0.1:6333/collections/document_chunks
```

Confirm `status: green`, vector `size: 768`, distance `Cosine`, and a positive
`points_count`. `indexed_vectors_count: 0` is normal below Qdrant's indexing
threshold.

### 2. Test retrieval

```bash
curl -X POST http://127.0.0.1:8000/api/v1/retrieval/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What technologies are mentioned?",
    "top_k": 3,
    "score_threshold": 0.2
  }'
```

The response should contain relevant chunks ordered from highest to lowest
cosine score. Raising the threshold can produce no matches; lowering it can
include loosely related chunks.

### 3. Test grounded chat

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What technologies are mentioned in the documents?",
    "conversation_id": null
  }'
```

The response should include a UUID `conversation_id`, `action: "answer"`, an
answer, and source records. Initial inference can be slow while Ollama loads
the model.

### 4. Test multi-turn Redis memory

Copy the returned conversation ID:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Which of those are used for mobile development?",
    "conversation_id": "PASTE-CONVERSATION-ID-HERE"
  }'
```

The follow-up should understand “those”. Verify two user and two assistant
messages were stored:

```bash
docker compose exec redis redis-cli LLEN chat:PASTE-CONVERSATION-ID-HERE:messages
```

Expected: `4`. Conversations expire after `CHAT_MEMORY_TTL_SECONDS` (24 hours
by default), and only the latest `CHAT_MEMORY_MAX_MESSAGES` are retained.

### 5. Test multi-turn interview booking

First send incomplete information:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to book an interview",
    "conversation_id": null
  }'
```

Expect `"action":"booking_details_required"`. Copy its conversation ID and
use a future date in the follow-up:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My name is Alex Doe, email alex@example.com, date 2027-01-15 at 10:30 AM",
    "conversation_id": "PASTE-BOOKING-CONVERSATION-ID-HERE"
  }'
```

Expect `"action":"interview_booked"` and a populated `booking` object.
Submitting the same date and time again returns HTTP `409`.

## Access PostgreSQL with Adminer

Open `http://localhost:9000`:

| Field | Value |
|---|---|
| System | PostgreSQL |
| Server | `postgres` |
| Username | `postgres` |
| Password | `postgres` |
| Database | `rag_backend` |

Use `postgres`, not `localhost`, as Adminer's server. Adminer is inside Docker,
so its `localhost` refers to itself.

Query bookings from the terminal:

```bash
docker compose exec postgres psql -U postgres -d rag_backend -c \
  "SELECT id, name, email, scheduled_date, scheduled_time FROM interview_bookings;"
```

## Everyday commands

```bash
# Start all infrastructure
docker compose up -d

# Start FastAPI
uv run uvicorn rag_backend.main:app --reload

# Show containers
docker compose ps

# Run tests
uv run pytest -q

# Stop containers but preserve data
docker compose down
```

After changing a SQLAlchemy model:

```bash
uv run alembic revision --autogenerate -m "describe the schema change"
uv run alembic upgrade head
```

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `DATABASE_URL` | PostgreSQL port 5433 | Async SQLAlchemy connection |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum upload size |
| `RECURSIVE_CHUNK_SIZE` | `1500` | Recursive chunk size |
| `RECURSIVE_CHUNK_OVERLAP` | `200` | Recursive overlap |
| `SEMANTIC_BREAKPOINT_PERCENTILE` | `80` | Semantic split sensitivity |
| `SEMANTIC_MIN_CHUNK_SIZE` | `400` | Semantic minimum target |
| `SEMANTIC_MAX_CHUNK_SIZE` | `1500` | Semantic hard maximum |
| `OLLAMA_URL` | port 11434 | Ollama API |
| `EMBEDDING_MODEL` | `embeddinggemma` | Embedding model |
| `EMBEDDING_BATCH_SIZE` | `32` | Embeddings per batch |
| `QDRANT_URL` | port 6333 | Qdrant API |
| `QDRANT_COLLECTION` | `document_chunks` | Vector collection |
| `CHAT_MODEL` | `gemma3:1b` | Generative model |
| `CHAT_TEMPERATURE` | `0.1` | Generation variability |
| `RETRIEVAL_TOP_K` | `5` | Maximum RAG sources |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.2` | Minimum cosine score |
| `REDIS_URL` | port 6379, database 0 | Redis connection |
| `CHAT_MEMORY_TTL_SECONDS` | `86400` | Conversation lifetime |
| `CHAT_MEMORY_MAX_MESSAGES` | `12` | Retained messages |

## Troubleshooting

### Docker API/named-pipe error on Windows

Start Docker Desktop, wait for the engine, and verify:

```powershell
docker version
wsl -l -v
```

The `docker-desktop` WSL distribution should run. Confirm Docker Desktop uses
Linux containers.

### `no matching manifest for windows/amd64`

Docker is using Windows containers. Switch Docker Desktop to Linux containers.

### A port is already in use

Check `5433`, `6333`, `6379`, `9000`, and `11434`. Stop the conflicting
program, or change the left side of the mapping in `docker-compose.yml` and the
matching `.env` URL. For example, `5433:5432` maps host `5433` to container
`5432`.

### PostgreSQL password authentication failed

Confirm `.env` contains:

```env
DATABASE_URL="postgresql+psycopg://postgres:postgres@127.0.0.1:5433/rag_backend"
```

PostgreSQL credentials initialize only when its volume is first created.
Changing Compose variables does not change an existing database password.

### Adminer cannot connect

Use `postgres` as the server. Do not use `localhost` from Adminer.

### Ollama model not found

```bash
docker compose exec ollama ollama list
docker compose exec ollama ollama pull embeddinggemma
docker compose exec ollama ollama pull gemma3:1b
```

### Duplicate upload returns HTTP 409

Upload a different file. Renaming identical content does not change its hash.

### FastAPI cannot import `rag_backend`

Run from the repository root:

```bash
uv run uvicorn rag_backend.main:app --reload
```

Do not run Uvicorn from inside `src/rag_backend`.

### Completely reset Docker data

Only when a destructive reset is intentional:

```bash
docker compose down -v
```

**Warning:** this permanently deletes PostgreSQL data, Qdrant vectors, Redis
data, and downloaded Ollama models. Models must be downloaded again.

## Submission checklist

- [ ] `.env` is not committed.
- [ ] `docker compose ps` shows all services running.
- [ ] Both Ollama models are installed.
- [ ] Alembic reports `b4172e18eac1 (head)`.
- [ ] Tests pass and Alembic detects no missing migration.
- [ ] Recursive and semantic uploads complete.
- [ ] Qdrant has stored points.
- [ ] Retrieval returns relevant chunks.
- [ ] Chat returns a grounded answer and sources.
- [ ] A follow-up works with the same conversation ID.
- [ ] An interview booking appears in PostgreSQL.
- [ ] `git status` is clean before submission.
