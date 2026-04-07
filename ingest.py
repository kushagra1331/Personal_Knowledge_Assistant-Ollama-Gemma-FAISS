import os

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import *
from db import fetch_document_by_hash, fetch_documents_by_ids, init_db, insert_document


def _build_chunk_header(metadata, created_at=None):
    header_lines = [
        f"Title: {metadata.get('title', 'Untitled')}",
        f"Tags: {', '.join(metadata.get('tags', []))}",
        f"Source type: {metadata.get('source_type', 'unknown')}",
        f"Source value: {metadata.get('source_value', 'unknown')}",
    ]
    if created_at:
        header_lines.append(f"Saved at: {created_at}")
    if metadata.get("document_date"):
        header_lines.append(f"Document date: {metadata['document_date']}")
    if metadata.get("summary"):
        header_lines.append(f"Document summary: {metadata['summary']}")
    return "\n".join(header_lines)


def ingest_documents(documents):
    init_db()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)

    chunks = []
    for doc in documents:
        content_hash = doc.metadata.get("content_hash")
        existing_doc = fetch_document_by_hash(content_hash)
        if existing_doc:
            continue

        doc_id = insert_document(
            doc.metadata["title"],
            doc.page_content,
            doc.metadata["tags"],
            source_type=doc.metadata.get("source_type"),
            source_value=doc.metadata.get("source_value"),
            summary=doc.metadata.get("summary"),
            document_date=doc.metadata.get("document_date"),
            content_hash=content_hash,
        )
        stored_doc = fetch_documents_by_ids([doc_id])[0]

        enriched_doc = doc.model_copy(
            update={
                "metadata": {
                    **doc.metadata,
                    "doc_id": doc_id,
                    "created_at": stored_doc.get("created_at"),
                }
            }
        )
        split_docs = splitter.split_documents([enriched_doc])
        for chunk in split_docs:
            chunk_text = chunk.page_content
            chunk.metadata["chunk_text"] = chunk_text
            chunk_header = _build_chunk_header(
                chunk.metadata,
                created_at=chunk.metadata.get("created_at"),
            )
            chunk.page_content = f"{chunk_header}\n\nContent:\n{chunk_text}"
            chunks.append(chunk)

    if not chunks:
        return

    if os.path.exists(FAISS_INDEX_DIR):
        vs = FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings)

    vs.save_local(FAISS_INDEX_DIR)
