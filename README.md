# Customer Support RAG MVP

A lightweight, full-stack Retrieval-Augmented Generation (RAG) application for building a customer support assistant over your own documents and public help pages.

The project runs as a single FastAPI application. The customer chat widget and admin dashboard are built with plain HTML, CSS, and JavaScript and are served directly by FastAPI. No React, Vue, Next.js, Node.js, or separate frontend process is required.

## Features

- Floating customer support chat widget
- Persian and mixed RTL/LTR text support
- Minimal admin dashboard at `/admin`
- Upload support for TXT, PDF, and JSON files
- Bounded same-origin web crawler for public documentation pages
- Local multilingual embeddings with SentenceTransformers
- Persistent vector search with ChromaDB
- Document and ingestion metadata stored in SQLite
- GapGPT integration through its OpenAI-compatible API
- Conversation history and source citations
- Deterministic document and vector identifiers
- Idempotent document ingestion
- Embedding fingerprint tracking to prevent incompatible vectors from being mixed
- Bearer-token authentication for administrative endpoints
- Configurable upload, crawl, PDF, text, and conversation limits
- Docker Compose support
- Automated backend tests and CI configuration

## Technology Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI |
| Chat model | GapGPT through the OpenAI-compatible API |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector database | ChromaDB |
| Metadata database | SQLite |
| Frontend | Vanilla HTML, CSS, and JavaScript |
| Document parsing | PyPDF, Python JSON and text parsers |
| Web crawling | HTTPX and BeautifulSoup |
| Testing | Pytest |
| Packaging | `pyproject.toml` and `uv.lock` |

## Architecture

```text
Customer question
       |
       v
Local multilingual query embedding
       |
       v
ChromaDB vector search
       |
       v
Active document filtering through SQLite
       |
       v
Relevant support excerpts + conversation history
       |
       v
GapGPT chat completion
       |
       v
Grounded answer + source citations
```

Document ingestion follows this flow:

```text
TXT / PDF / JSON / crawled HTML
       |
       v
Validation and text extraction
       |
       v
Overlapping text chunks
       |
       v
Local multilingual embeddings
       |
       +----------> ChromaDB vectors
       |
       +----------> SQLite document and chunk metadata
```

## Project Structure

```text
.
├── app/
│   ├── llm/
│   │   └── provider.py          # GapGPT chat and local embedding providers
│   ├── rag/
│   │   ├── chunking.py          # Overlapping text chunking
│   │   ├── ingestion.py         # TXT, PDF, and JSON parsing
│   │   ├── prompting.py         # Grounded customer-support prompt
│   │   └── vectorstore.py       # Persistent ChromaDB wrapper
│   ├── services/
│   │   ├── chat.py              # Retrieval and answer orchestration
│   │   ├── crawler.py           # Bounded public web crawler
│   │   ├── documents.py         # Document indexing and replacement
│   │   └── metadata.py          # SQLite metadata repository
│   ├── static/
│   │   ├── index.html           # Customer-facing page and chat widget
│   │   ├── admin.html           # Admin dashboard
│   │   ├── api.js               # Browser API client
│   │   └── styles.css           # Shared responsive styling
│   ├── config.py                # Environment-based settings
│   ├── main.py                  # FastAPI app factory and routes
│   └── schemas.py               # API request and response schemas
├── scripts/
│   ├── ingest.py                # Upload a directory through the admin API
│   └── smoke_test.py            # Basic API smoke test
├── tests/                       # Unit and API tests
├── .env.example                 # Environment template
├── Dockerfile.backend
├── compose.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

## Prerequisites

### Local development

- Python 3.10 or newer
- Internet access during the first run to download the embedding model
- A valid GapGPT API key

### Docker

- Docker Desktop or Docker Engine
- Docker Compose
- A valid GapGPT API key
- Internet access during the first container startup to download the embedding model

Node.js and npm are not required.

## GapGPT API Key

1. Open the GapGPT developer dashboard.
2. Go to the API Keys section.
3. Create a new API key.
4. Copy the key and store it securely.

The key is shown only once. Do not commit it to Git and do not place it in frontend JavaScript.

The application uses the following OpenAI-compatible endpoint:

```text
https://api.gapgpt.app/v1
```

## Environment Configuration

Create your local environment file from the provided template.

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### Linux or macOS

```bash
cp .env.example .env
```

Open `.env` and configure it:

```env
# Application
APP_ENV=development
ASSISTANT_NAME=Support Assistant
COMPANY_NAME=Your Company
DATA_DIR=app/data/runtime
SQLITE_PATH=app/data/runtime/rag.sqlite3
CHROMA_PERSIST_DIR=app/data/runtime/chroma
CHROMA_COLLECTION_NAME=customer_support_multilingual_minilm_normalized_paragraph_v2

# GapGPT chat provider
CHAT_PROVIDER=gapgpt
CHAT_MODEL=gpt-4o
CHAT_BASE_URL=https://api.gapgpt.app/v1
CHAT_API_KEY=YOUR_REAL_GAPGPT_API_KEY

# Local multilingual embeddings
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_BATCH_SIZE=64

# Retrieval and chunking
TOP_K=4
RETRIEVAL_MAX_DISTANCE=0.65
CHUNK_SIZE=900
CHUNK_OVERLAP=150
MAX_HISTORY_MESSAGES=12
MAX_MESSAGE_CHARS=4000

# Admin authentication
ADMIN_API_KEY=replace-this-with-a-long-random-secret

# Optional direct browser origins
# The bundled frontend is served by FastAPI and does not require CORS.
CORS_ALLOWED_ORIGINS=

# Ingestion limits
MAX_UPLOAD_BYTES=10485760
MAX_EXTRACTED_CHARS=1000000
MAX_PDF_PAGES=200
MAX_CRAWL_PAGES=5
MAX_CRAWL_DEPTH=1
MAX_CRAWL_RESPONSE_BYTES=2097152
CRAWL_TIMEOUT_SECONDS=15
```

### Important secrets

Replace these values:

```env
CHAT_API_KEY=YOUR_REAL_GAPGPT_API_KEY
ADMIN_API_KEY=replace-this-with-a-long-random-secret
```

`CHAT_API_KEY` is sent only by the backend to GapGPT.

`ASSISTANT_NAME` and `COMPANY_NAME` control the public chat branding and the assistant identity used in responses. For example:

```env
ASSISTANT_NAME=Jibi Assistant
COMPANY_NAME=Jibi
```

`RETRIEVAL_MAX_DISTANCE` controls the maximum accepted cosine distance for retrieved chunks. Lower values are stricter. The default `0.65` is a starting point and should be evaluated against your own documentation.

`ADMIN_API_KEY` is the password used to unlock the admin dashboard. For example:

```env
ADMIN_API_KEY=my-local-admin-key
```

You would then enter the following value on the `/admin` login screen:

```text
my-local-admin-key
```

After changing `.env`, restart the FastAPI process. Environment changes are not guaranteed to be applied to an already-running process.

## Local Installation and Running

There are two supported local installation methods.

### Option 1: Using uv (recommended)

Open PowerShell or a terminal in the repository root:

```powershell
Set-Location "D:\Project\Basalam\RAG"
```

Install `uv` if it is not already available:

```powershell
python -m pip install uv
```

Install the locked project dependencies and test tools:

```powershell
uv sync --extra test
```

Create and configure `.env`:

```powershell
Copy-Item .env.example .env
```

Start the application:

```powershell
uv run uvicorn app.main:app --reload
```

### Option 2: Using pip and a virtual environment

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, allow it for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

Install the application and test dependencies:

```powershell
python -m pip install --upgrade pip
pip install -e ".[test]"
```

Start the application:

```powershell
python -m uvicorn app.main:app --reload
```

## First Startup

The first application startup initializes the local multilingual embedding model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

The model is downloaded from Hugging Face and cached locally. The first startup may therefore take several minutes, depending on the network connection and machine performance.

Later startups reuse the cached files and should be faster.

A successful startup looks similar to:

```text
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal open while using the application.

Stop the application with:

```text
Ctrl + C
```

## Application URLs

After the server starts, open:

| Page | URL |
|---|---|
| Customer page and chat widget | `http://127.0.0.1:8000/` |
| Admin dashboard | `http://127.0.0.1:8000/admin` |
| Health check | `http://127.0.0.1:8000/healthz` |
| Swagger API documentation | `http://127.0.0.1:8000/docs` |
| OpenAPI schema | `http://127.0.0.1:8000/openapi.json` |

Do not open `app/static/index.html` directly from the filesystem. The frontend must be opened through FastAPI so that its CSS, JavaScript, and API requests use the correct URLs.

## Testing the Application Manually

### 1. Check the health endpoint

Open:

```text
http://127.0.0.1:8000/healthz
```

Expected response:

```json
{
  "status": "ok"
}
```

PowerShell alternative:

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/healthz"
```

### 2. Open the admin dashboard

Open:

```text
http://127.0.0.1:8000/admin
```

Enter the exact value configured in `.env`:

```env
ADMIN_API_KEY=my-local-admin-key
```

In this example, enter:

```text
my-local-admin-key
```

The key is stored only in the current browser tab's `sessionStorage`. It is removed when the tab is closed or when you click Log Out.

### 3. Create a sample support document

Create a file named `faq.txt` in the project directory with this content:

```text
Return Policy

Customers may return a product within seven days after delivery.
The product must be unused, undamaged, and returned in its original packaging.
If the product is defective, the store pays the return shipping cost.
To start a return request, contact customer support.
```

PowerShell command:

```powershell
@'
Return Policy

Customers may return a product within seven days after delivery.
The product must be unused, undamaged, and returned in its original packaging.
If the product is defective, the store pays the return shipping cost.
To start a return request, contact customer support.
'@ | Set-Content -Path "faq.txt" -Encoding utf8
```

A Persian test document can also be used:

```powershell
@'
شرایط مرجوعی کالا

مشتری می‌تواند کالا را تا هفت روز پس از تحویل مرجوع کند.
کالا باید سالم، استفاده‌نشده و دارای بسته‌بندی اصلی باشد.
اگر کالا معیوب باشد، هزینه ارسال مرجوعی بر عهده فروشگاه است.
برای شروع فرایند مرجوعی با پشتیبانی تماس بگیرید.
'@ | Set-Content -Path "faq-fa.txt" -Encoding utf8
```

### 4. Upload the document

In the admin dashboard:

1. Find the file upload panel.
2. Click the file selection button or drag the file into the dropzone.
3. Select `faq.txt` or `faq-fa.txt`.
4. Wait for ingestion to finish.
5. Confirm that the document appears in the document list.
6. Confirm that the active-document and indexed-chunk counters increase.

During ingestion, the application:

1. Validates the file type and size.
2. Extracts and normalizes text.
3. Splits text into overlapping chunks.
4. Generates local multilingual embeddings.
5. Stores vectors in ChromaDB.
6. Stores document, chunk, and ingestion metadata in SQLite.

### 5. Test the customer chat

Open:

```text
http://127.0.0.1:8000/
```

Click the floating chat button and ask:

```text
How many days do I have to return a product?
```

Or in Persian:

```text
چند روز برای مرجوع کردن کالا فرصت دارم؟
```

The expected response should mention seven days and display a source citation referring to the uploaded file.

Test a second question:

```text
Who pays the return shipping cost for a defective product?
```

The answer should be grounded in the uploaded document.

### 6. Test the no-answer behavior

Ask a question that is not covered by the uploaded documents:

```text
What will the weather be tomorrow?
```

The assistant should state that it cannot answer from the available support documentation. It should not invent an answer.

## Testing Admin Authentication from PowerShell

Configure a matching key in `.env`:

```env
ADMIN_API_KEY=my-local-admin-key
```

Call the status endpoint:

```powershell
$headers = @{
    Authorization = "Bearer my-local-admin-key"
}

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/admin/status" `
  -Headers $headers
```

Expected fields include:

```text
status
active_documents
indexed_chunks
chat_provider
embedding_provider
recent_runs
```

A missing or incorrect key returns HTTP `401`.

## Uploading a Directory from the Command Line

The included ingestion script uploads all supported files in a directory through the authenticated admin API.

Set the admin key:

```powershell
$env:ADMIN_API_KEY="my-local-admin-key"
```

Upload supported files from `app/data/raw`:

```powershell
python scripts/ingest.py app/data/raw
```

Or provide another directory:

```powershell
python scripts/ingest.py "D:\SupportDocuments"
```

Supported formats:

- `.txt`
- `.pdf`
- `.json`

## Web Crawler

The admin dashboard can ingest public help pages.

1. Open `/admin`.
2. Enter a public HTTP or HTTPS URL.
3. Choose the maximum page count and crawl depth.
4. Start the crawl.
5. Wait for the resulting pages to appear in the document list.

The crawler intentionally applies restrictions:

- Only absolute HTTP and HTTPS URLs are accepted.
- Only default HTTP/HTTPS ports are accepted.
- Localhost, private, link-local, and reserved IP addresses are rejected.
- Only same-origin links are followed.
- Response size, page count, depth, and timeout are limited.
- Non-HTML resources are ignored.
- Script, style, navigation, header, and footer content is removed.

Do not use the crawler for private dashboards, authenticated sites, intranet hosts, or internal cloud metadata endpoints.

## Changing the Embedding Model

Vectors produced by different embedding models must never be mixed, even when their dimensions are identical.

The current default collection is:

```env
CHROMA_COLLECTION_NAME=customer_support_multilingual_minilm_normalized_paragraph_v2
```

If you change `EMBEDDING_MODEL`:

1. Change `CHROMA_COLLECTION_NAME` to a new versioned name.
2. Restart the application.
3. Re-upload or re-crawl all documents.

For example:

```env
EMBEDDING_MODEL=some-other-model
CHROMA_COLLECTION_NAME=customer_support_embedding_v2
```

The application stores an embedding fingerprint in SQLite and Chroma metadata. An unchanged document is skipped only when both its content hash and embedding fingerprint match.

Old Chroma collections are not automatically queried or merged with the new collection.

## Clearing Local Development Data

Stop the application first.

Remove the runtime data directory:

```powershell
Remove-Item "app\data\runtime" -Recurse -Force
```

Then restart the application and upload the documents again.

This deletes:

- SQLite metadata
- ChromaDB vectors
- ingestion history

It does not delete the downloaded SentenceTransformers model cache.

## Running Automated Tests

### With uv

```powershell
uv run pytest
```

Run lint checks:

```powershell
uv run ruff check app tests scripts
```

Apply formatting:

```powershell
uv run ruff format app tests scripts
```

### With an activated virtual environment

```powershell
pytest
ruff check app tests scripts
ruff format app tests scripts
```

The test suite uses fake and mocked embedding providers. Running tests does not download the production embedding model.

Current expected result:

```text
20 passed
```

## Running with Docker Compose

Create `.env` first:

```powershell
Copy-Item .env.example .env
```

Add your real GapGPT key and a secure admin key.

Build and start the service:

```powershell
docker compose up --build
```

Open:

```text
http://localhost:8000
```

The Compose setup uses two persistent volumes:

| Volume | Purpose |
|---|---|
| `rag_data` | SQLite database and ChromaDB vectors |
| `model_cache` | Hugging Face and SentenceTransformers model cache |

The first container startup may take longer while the model is downloaded. Later containers reuse `model_cache`.

Run in the background:

```powershell
docker compose up --build -d
```

View logs:

```powershell
docker compose logs -f app
```

Stop the service without deleting data:

```powershell
docker compose down
```

Stop the service and delete all persistent data and model cache:

```powershell
docker compose down -v
```

## API Endpoints

### Public endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Customer page and chat widget |
| `GET` | `/healthz` | Service health check |
| `POST` | `/api/chat` | RAG chat request |

### Admin endpoints

All admin endpoints require:

```text
Authorization: Bearer YOUR_ADMIN_API_KEY
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin` | Admin dashboard |
| `GET` | `/api/admin/status` | Service and ingestion status |
| `GET` | `/api/admin/documents` | List indexed documents |
| `POST` | `/api/admin/upload` | Upload and index one document |
| `POST` | `/api/admin/crawl` | Crawl and index public pages |

### Chat request example

```json
{
  "message": "What is the return policy?",
  "history": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help?"
    }
  ]
}
```

### Chat response example

```json
{
  "answer": "Customers may return a product within seven days after delivery. [S1]",
  "sources": [
    {
      "id": "S1",
      "document_id": "document-uuid",
      "title": "faq.txt",
      "source_type": "upload",
      "url": null,
      "chunk_index": 0
    }
  ]
}
```

## Troubleshooting

### The admin key is rejected

Make sure the value entered in the browser exactly matches `.env`:

```env
ADMIN_API_KEY=my-local-admin-key
```

Restart the server after changing `.env`.

### The page has no styling

Do not open the HTML files directly. Open the page through FastAPI:

```text
http://127.0.0.1:8000/
```

Force-refresh the browser:

```text
Ctrl + Shift + R
```

Verify that the stylesheet is available:

```text
http://127.0.0.1:8000/static/styles.css
```

### GapGPT returns an authentication error

Verify:

```env
CHAT_PROVIDER=gapgpt
CHAT_BASE_URL=https://api.gapgpt.app/v1
CHAT_API_KEY=YOUR_REAL_GAPGPT_API_KEY
```

Make sure the key is active and the GapGPT wallet has sufficient credit.

### The embedding model download is slow

The first run downloads the multilingual model from Hugging Face. Keep the process running until initialization finishes.

On Windows, Hugging Face may display a symlink warning. The cache still works, but may consume more disk space. Enabling Windows Developer Mode allows symlink-based caching.

### ChromaDB reports a vector dimension mismatch

The selected Chroma collection contains vectors from another embedding model.

Use a new collection name:

```env
CHROMA_COLLECTION_NAME=customer_support_multilingual_minilm_normalized_paragraph_v3
```

Then restart and re-ingest all documents.

### Port 8000 is already in use

Run on another port:

```powershell
python -m uvicorn app.main:app --reload --port 8001
```

Open:

```text
http://127.0.0.1:8001/
```

### Reset the browser chat history

Use the reset button inside the chat widget, or clear the site's local storage in browser developer tools.

## Security Notes

- Never commit `.env`.
- Never place the GapGPT key in frontend JavaScript.
- Use a long random `ADMIN_API_KEY` in deployed environments.
- The public chat can retrieve all indexed knowledge-base content. Do not upload private or tenant-specific information to this MVP.
- Apply request-body and rate limits at a reverse proxy before exposing the service publicly.
- Use HTTPS in production.
- Review crawler restrictions before allowing untrusted administrators to submit URLs.
- Run a single application worker for this synchronous MVP unless ingestion coordination is moved to a process-shared system.

## MVP Limitations

- No user accounts or role-based access control
- No background queue
- No ticketing or human handoff integration
- No conversation database
- No streaming chat responses
- No multi-tenant document isolation
- No distributed ingestion lock
- Local ChromaDB is intended for a single application instance

These constraints intentionally keep the MVP easy to install, understand, and run.

## License

Add a `LICENSE` file before distributing the project as open-source software.
