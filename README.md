# 🚀 Personal Knowledge Assistant (Local RAG with Gemma 4 + Ollama)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LLM](https://img.shields.io/badge/LLM-Gemma%204-orange)
![Ollama](https://img.shields.io/badge/Runtime-Ollama-green)
![VectorDB](https://img.shields.io/badge/VectorDB-FAISS-red)
![Status](https://img.shields.io/badge/Status-Active-success)

> 🔒 **100% Offline Personal AI Assistant powered by Gemma 4 via Ollama**  
> ⚡ Build, search, and query your own knowledge base using a local RAG pipeline

---

## 🧠 What is this?

**Personal Knowledge Assistant** is a **local-first Retrieval-Augmented Generation (RAG) system** that allows you to:

- Build your own knowledge base from **PDFs, URLs, and raw text**
- Run **LLM inference locally using Gemma 4 via Ollama**
- Perform **semantic search with FAISS**
- Store and manage documents with **SQLite metadata**
- Ask questions and get **context-aware answers grounded in your data**

👉 No OpenAI. No external APIs. Fully private.

---

## 🎥 Demo

![Demo](Assets/demo.gif)

---

## 🔑 Key Highlights

- ⚡ **Gemma 4 + Ollama (Local LLM + Embeddings)**  
  Fully offline inference with zero API cost

- 🧠 **RAG Pipeline**  
  Combines retrieval + generation for accurate, context-aware answers

- 🔍 **FAISS Vector Search**  
  Fast and efficient semantic similarity search

- 🗄️ **SQLite Metadata Layer**  
  Enables filtering, tagging, and structured retrieval

- 📂 **Multi-Source Ingestion**
  - PDFs  
  - URLs  
  - Raw text  

- 💻 **Streamlit UI**
  - Add knowledge  
  - Ask questions  
  - Browse saved data  

---

## ⚙️ Tech Stack

| Layer              | Technology |
|------------------|-----------|
| LLM              | **Gemma 4 (via Ollama)** |
| Embeddings       | **nomic-embed-text (Ollama)** |
| Vector DB        | **FAISS** |
| Metadata Store   | **SQLite** |
| Orchestration    | **LangChain** |
| UI               | **Streamlit** |
| Extraction       | PyPDF, BeautifulSoup |

---

## 🏗️ Architecture

User Input (PDF / URL / Text)  
↓  
Content Extraction + Cleaning  
↓  
Metadata Generation (tags, summary, timestamps)  
↓  
SQLite Storage (documents + metadata)  
↓  
Chunking + Embedding (Ollama)  
↓  
FAISS Index (vector storage)  
↓  
User Query  
↓  
Metadata Filtering (SQLite)  
↓  
Semantic Retrieval (FAISS)  
↓  
Context → Gemma 4 (Ollama)  
↓  
Final Answer  

---

## 📦 Features

- ✅ Fully **local RAG pipeline**
- ✅ **Gemma 4-powered reasoning**
- ✅ No external API dependency
- ✅ Metadata-aware retrieval (not just vector similarity)
- ✅ Tagging + structured document tracking
- ✅ Streamlit-based interactive UI
- ✅ Persistent storage via SQLite
- ✅ Fast similarity search with FAISS

---

## 📁 Project Structure

app.py              # Streamlit UI  
rag.py              # Retrieval + answer generation  
ingest.py           # Document ingestion pipeline  
db.py               # SQLite schema + helpers  
extractors.py       # PDF, URL, text extraction  
config.py           # Environment configuration  
utils/helpers.py    # Metadata + utilities  

---

## ⚡ Getting Started

### 1. Clone Repo

git clone <your-repo-url>  
cd knowledge-copilot-full-updated  

---

### 2. Setup Environment

python -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

---

### 3. Setup Ollama + Models

ollama pull gemma4  
ollama pull nomic-embed-text  

---

### 4. Configure Environment

Create `.env`:

OLLAMA_MODEL=gemma4  
OLLAMA_BASE_URL=http://localhost:11434  
OLLAMA_EMBED_MODEL=nomic-embed-text  
FAISS_INDEX_DIR=data/faiss_index  
SQLITE_DB_PATH=data/knowledge.db  
CHUNK_SIZE=1000  
CHUNK_OVERLAP=150  
TOP_K=4  

---

### 5. Run the App

streamlit run app.py  

👉 Open: http://localhost:8501  

---

## 🧑‍💻 How to Use

### ➕ Add Knowledge
- Upload PDF, URL, or raw text  
- Add tags for better retrieval  

### ❓ Ask Questions
Examples:
- Summarize what I uploaded today  
- What does my water bill say?  
- Compare my exercise and diet notes  

### 📂 Browse
- View saved documents  
- Inspect metadata and summaries  
- Filter using tags  

---

## 🔮 Future Improvements

- OCR support for scanned PDFs  
- Reranking for improved retrieval accuracy  
- Chat memory / session context  
- Document collections / folders  
- Highlighted citations in responses  

