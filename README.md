# RAG Demo

A local Retrieval-Augmented Generation (RAG) demo for asking questions over your own files and documents. The project uses FastAPI for the backend, Streamlit for the chat UI, ChromaDB as the vector database, and Sentence Transformers for embeddings.

## Features

- Load raw documents and split them into searchable chunks
- Generate embeddings and persist them in ChromaDB
- Retrieve the most relevant context for each user question
- Generate answers with an OpenAI-compatible LLM provider such as GapGPT
- Simple FastAPI backend
- Streamlit chat interface with RTL/Persian-friendly rendering
- Basic tests for chunking, retrieval, pipeline behavior, and API endpoints

## Project Structure

```text
.
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Environment-based configuration
│   ├── schemas.py              # Request and response models
│   ├── llm/                    # LLM provider integration
│   ├── rag/                    # Ingestion, chunking, retrieval, and generation
│   ├── ui/streamlit_app.py     # Streamlit chat UI
│   └── data/                   # Raw documents and ChromaDB data
├── scripts/
│   ├── ingest.py               # Trigger ingestion through the API
│   └── smoke_test.py           # Manual smoke test script
├── tests/
├── .env.example
├── requirements.txt
└── pyproject.toml
```

## Requirements

- Python 3.10 or later
- An API key for your LLM provider

## Installation

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/Ehsan24Elyasi/RAG.git
cd RAG
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Variables

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Then update the values in `.env`:

```env
LLM_PROVIDER=gapgpt
LLM_API_KEY=YOUR_GAPGPT_API_KEY
LLM_MODEL=gpt-4o
LLM_BASE_URL=https://api.gapgpt.app/v1
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=app/data/chroma
RAW_DATA_DIR=app/data/raw
TOP_K=4
CHUNK_SIZE=700
CHUNK_OVERLAP=100
```

> Security note: `.env` contains sensitive credentials and must not be committed to GitHub. It is already listed in `.gitignore`.

## Add Documents

Put your text or PDF files in:

```text
app/data/raw
```

Create the directory if it does not already exist.

## Run the Backend

```powershell
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Health check endpoint:

```text
GET /health
```

## Ingest Documents

After the backend is running, start ingestion:

```powershell
python scripts/ingest.py
```

You can also call the endpoint directly:

```text
POST /ingest
```

## Run the Streamlit UI

In another terminal, activate the virtual environment and run:

```powershell
streamlit run app/ui/streamlit_app.py
```

Open the Streamlit URL in your browser and start asking questions.

## API Example

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/query" `
  -ContentType "application/json" `
  -Body '{"question":"What is this document about?","debug":false}'
```

Example response:

```json
{
  "answer": "...",
  "sources": [
    {
      "file_name": "example.pdf",
      "chunk_index": 0
    }
  ],
  "retrieved_context": null
}
```

## Run Tests

```powershell
pytest
```

## GitHub Notes

Before pushing the project to GitHub, make sure sensitive or large generated files are not committed. The following paths are ignored:

- `.env`
- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `app/data/chroma/`
- `app/data/processed/`

## License

Add a `LICENSE` file if you want to publish this project with an explicit open-source license.
