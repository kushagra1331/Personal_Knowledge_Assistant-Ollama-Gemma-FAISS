# Personal Knowledge Assistant

Personal Knowledge Assistant is a local Retrieval-Augmented Generation (RAG) application that lets you build a personal knowledge base from URLs, PDFs, and raw text, then ask questions over that data through a Streamlit interface.

The project is designed for local-first experimentation. It uses Ollama for local LLM and embedding inference, FAISS for vector similarity search, and SQLite for document storage and metadata. The result is a lightweight personal knowledge assistant that can ingest your own documents and answer questions using the information you saved.

## Demo

![Demo GIF](Assets/demo.gif)


## What This Project Does

With this app, a user can:

- ingest content from a web URL
- ingest pasted raw text
- ingest a PDF document
- store document content and metadata in SQLite
- create embeddings and save them in a FAISS vector index
- ask questions over the saved knowledge base
- browse previously saved items inside the Streamlit UI

The app also stores richer metadata for each document, such as:

- source type
- source value
- tags
- ingestion timestamp
- lightweight summary
- extracted document date when available

This makes retrieval more useful than plain chunk similarity alone, because the system can combine metadata-aware filtering with semantic search.

## Tech Stack

This project uses the following technologies:

- `Streamlit` for the web app UI
- `Ollama` for local LLM inference and embeddings
- `LangChain` for orchestration around document handling and vector retrieval
- `FAISS` for vector storage and similarity search
- `SQLite` for document persistence and metadata
- `PyPDF` for PDF text extraction
- `Requests` and `BeautifulSoup` for URL content extraction
- `python-dotenv` for environment-based configuration

## Architecture

At a high level, the application works like this:

1. A user uploads a URL, PDF, or raw text in the Streamlit app.
2. The content is extracted and normalized.
3. Document metadata is generated, including tags, source information, summary, hash, and possible document date.
4. The full document is stored in SQLite.
5. The document is split into chunks.
6. Each chunk is enriched with document metadata and embedded using an Ollama embedding model.
7. The embeddings are stored in FAISS.
8. When the user asks a question, the app first narrows candidates using SQLite metadata, then runs semantic retrieval through FAISS, and finally sends the selected context to the LLM for answering.

## Project Structure

Key files in this repository:

- [app.py](app.py): Streamlit UI
- [rag.py](rag.py): retrieval and answer generation logic
- [ingest.py](ingest.py): document ingestion and vector index updates
- [db.py](db.py): SQLite schema and database helpers
- [extractors.py](extractors.py): URL, PDF, and raw text extractors
- [config.py](config.py): environment-based configuration
- [utils/helpers.py](utils/helpers.py): metadata, summaries, and document-building helpers

## Features

- Local-first RAG pipeline
- No paid API required if you run Ollama locally
- URL, PDF, and raw text ingestion
- Metadata-aware retrieval
- Source display in answers
- Streamlit UI with tabs for ingestion, Q&A, and browsing
- SQLite-backed saved document history
- FAISS-backed fast semantic search

## Requirements

Before running the project, make sure you have:

- Python 3.10+ installed
- Ollama installed and running locally
- the required Ollama models pulled locally

This project currently expects:

- a chat model such as `gemma4`
- an embedding model such as `nomic-embed-text`

## Installation

Clone the repository and move into the project directory:

```bash
git clone <your-repo-url>
cd knowledge-copilot-full-updated
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Ollama Setup

Start Ollama on your machine, then pull the required models:

```bash
ollama pull gemma4
ollama pull nomic-embed-text
```

If you want to use different models, update your environment configuration accordingly.

## Environment Configuration

Create a `.env` file based on `.env.example`.

Example configuration:

```env
OLLAMA_MODEL=gemma4
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
FAISS_INDEX_DIR=data/faiss_index
SQLITE_DB_PATH=data/knowledge.db
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=4
```

What these values mean:

- `OLLAMA_MODEL`: the local chat model used to generate answers
- `OLLAMA_BASE_URL`: the local Ollama server URL
- `OLLAMA_EMBED_MODEL`: the embedding model used for vector search
- `FAISS_INDEX_DIR`: where the vector index is stored
- `SQLITE_DB_PATH`: where the SQLite database is stored
- `CHUNK_SIZE`: max chunk size for splitting documents
- `CHUNK_OVERLAP`: overlap between chunks
- `TOP_K`: how many top retrieved chunks/documents are considered during answering

## Running the App

Start the Streamlit app:

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

Important:

- make sure `which python` points to your project `.venv`, not Anaconda or system Python
- prefer `python -m pip ...` and `python -m streamlit ...` so the same interpreter is used for install and runtime
- if you renamed the project folder and `.venv` stops activating correctly, recreate the virtual environment so its activation scripts point at the new path

## How To Use

The UI is split into three top tabs:

### 1. Add Knowledge

Use this tab to ingest:

- a URL
- pasted raw text
- a PDF

You can also assign tags while uploading content.

### 2. Ask Questions

Use this tab to ask questions over your saved knowledge base.

Examples:

- `Summarize what I uploaded today`
- `What does the water bill say about the billing period?`
- `What did I save about Gemma 4?`
- `Compare the notes I uploaded on exercise and nutrition`

### 3. Browse Saved Items

Use this tab to:

- inspect saved documents
- view tags
- view saved timestamps
- inspect summaries and previews

## How Another User Can Replicate This Project

If someone else wants to reproduce the same project from scratch, they can follow this exact flow:

1. Clone the repository.
2. Create a Python virtual environment.
3. Install the Python dependencies from `requirements.txt`.
4. Install and start Ollama locally.
5. Pull the required Ollama models.
6. Create a `.env` file using `.env.example`.
7. Run `streamlit run app.py`.
8. Upload a few sample sources such as a PDF, a URL, and some pasted text.
9. Ask questions and verify that responses include relevant source information.

To reproduce the same behavior reliably, they should use the same:

- Python version
- Ollama models
- chunk size and overlap settings
- FAISS and SQLite paths

## Example Workflow

Here is a simple end-to-end test workflow:

1. Upload a web article.
2. Upload a PDF bill or report.
3. Paste a short note as raw text.
4. Ask: `Summarize everything I uploaded today`.
5. Ask a more targeted question like: `What is the billing period mentioned in the water bill?`
6. Check the source list shown under the answer.

## Why SQLite and FAISS Are Both Used

The project intentionally uses both SQLite and FAISS because they solve different problems:

- `SQLite` stores the full source documents and structured metadata
- `FAISS` stores vector embeddings for semantic similarity search

This hybrid setup is better than using only one of them:

- SQLite is better for filters, metadata, timestamps, and browsing
- FAISS is better for semantic retrieval across document chunks

## Current Limitations

- URL extraction is basic and may capture noisy page text on some websites
- PDF extraction quality depends on the structure of the PDF
- metadata extraction is still lightweight and heuristic-based
- retrieval quality depends heavily on the local embedding model
- there is no authentication or multi-user separation yet

## Ideas For Future Improvements

- add document deletion from the UI
- add source links directly in the answer panel
- support OCR for scanned PDFs
- add reranking for more accurate retrieval
- add document collections or folders
- add chat history and session memory
- support citations with excerpt highlighting

## Dependencies

Current Python dependencies from [requirements.txt](requirements.txt):

- `streamlit`
- `python-dotenv`
- `requests`
- `beautifulsoup4`
- `pypdf`
- `langchain`
- `langchain-core`
- `langchain-text-splitters`
- `langchain-ollama`
- `langchain-community`
- `faiss-cpu`

## License

Add the license of your choice here before publishing on GitHub.

## Author Notes

This project is a good example of a practical local RAG application: simple enough to understand end to end, but complete enough to demonstrate document ingestion, metadata storage, vector search, and question answering in one workflow.
