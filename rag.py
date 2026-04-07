import re
from datetime import datetime, timedelta

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import *
from db import fetch_documents, fetch_documents_by_ids


STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "for",
    "from",
    "i",
    "in",
    "information",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "show",
    "summarize",
    "summary",
    "the",
    "this",
    "to",
    "today",
    "uploaded",
    "what",
}


def _normalize_tags(raw_tags):
    if isinstance(raw_tags, str):
        return [t.strip() for t in raw_tags.split(",") if t.strip()]
    if isinstance(raw_tags, list):
        return [t.strip() for t in raw_tags if isinstance(t, str) and t.strip()]
    return []


def _normalize_source_type(source_type):
    if not source_type:
        return None
    mapping = {
        "pdf": "pdf",
        "url": "url",
        "website": "url",
        "web": "url",
        "text": "raw_text",
        "note": "raw_text",
        "notes": "raw_text",
        "raw": "raw_text",
    }
    return mapping.get(source_type)


def _extract_date_filter(query):
    lowered = query.lower()
    if "today" in lowered:
        return datetime.now().strftime("%Y-%m-%d")
    if "yesterday" in lowered:
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
    if match:
        return match.group(1)

    match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", query)
    if match:
        month, day, year = match.group(1).split("/")
        return f"{year}-{int(month):02d}-{int(day):02d}"

    return None


def _extract_source_type_filter(query):
    lowered = query.lower()
    for phrase in ["pdf", "website", "url", "web", "raw text", "text", "note", "notes"]:
        if phrase in lowered:
            return _normalize_source_type(phrase)
    return None


def _tokenize(text):
    return [
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _metadata_overlap_score(doc, query_tokens):
    haystacks = [
        doc.get("title", ""),
        doc.get("tags", ""),
        doc.get("summary", ""),
        doc.get("source_type", ""),
        doc.get("source_value", ""),
        doc.get("document_date", ""),
        doc.get("created_at", ""),
    ]
    text = " ".join(part for part in haystacks if part)
    lowered = text.lower()
    return sum(1 for token in query_tokens if token in lowered)


def _select_candidate_documents(query, tag_filter):
    created_on = _extract_date_filter(query)
    source_type = _extract_source_type_filter(query)
    documents = fetch_documents(
        tag=tag_filter,
        created_on=created_on,
        source_type=source_type,
        limit=1000,
    )

    if not documents and (created_on or source_type or tag_filter):
        documents = fetch_documents(limit=1000)

    query_tokens = _tokenize(query)
    ranked_docs = sorted(
        documents,
        key=lambda doc: (
            _metadata_overlap_score(doc, query_tokens),
            doc.get("created_at") or "",
            doc.get("id", 0),
        ),
        reverse=True,
    )

    if query_tokens:
        relevant_docs = [
            doc for doc in ranked_docs if _metadata_overlap_score(doc, query_tokens) > 0
        ]
        if relevant_docs:
            ranked_docs = relevant_docs

    return ranked_docs[:20]


def _search_documents(vs, query, allowed_doc_ids=None):
    k = max(TOP_K * 4, 12)
    try:
        results = vs.similarity_search_with_relevance_scores(query, k=k)
        docs = []
        for doc, score in results:
            if allowed_doc_ids and doc.metadata.get("doc_id") not in allowed_doc_ids:
                continue
            doc.metadata["_score"] = float(score)
            doc.metadata["_score_kind"] = "relevance"
            docs.append(doc)
        return docs
    except Exception:
        try:
            results = vs.similarity_search_with_score(query, k=k)
            docs = []
            for doc, score in results:
                if allowed_doc_ids and doc.metadata.get("doc_id") not in allowed_doc_ids:
                    continue
                doc.metadata["_score"] = float(score)
                doc.metadata["_score_kind"] = "distance"
                docs.append(doc)
            return docs
        except Exception:
            docs = vs.similarity_search(query, k=k)
            if allowed_doc_ids:
                docs = [doc for doc in docs if doc.metadata.get("doc_id") in allowed_doc_ids]
            return docs


def _is_better_score(candidate, current, score_kind):
    if candidate is None:
        return False
    if current is None:
        return True
    if score_kind == "distance":
        return candidate < current
    return candidate > current


def _dedupe_documents(docs):
    unique_docs = []
    seen_by_doc_id = {}

    for doc in docs:
        doc_id = doc.metadata.get("doc_id")
        if doc_id is None:
            unique_docs.append(doc)
            continue

        existing = seen_by_doc_id.get(doc_id)
        if existing is None:
            seen_by_doc_id[doc_id] = doc
            unique_docs.append(doc)
            continue

        candidate_score = doc.metadata.get("_score")
        current_score = existing.metadata.get("_score")
        score_kind = doc.metadata.get("_score_kind") or existing.metadata.get("_score_kind")
        if _is_better_score(candidate_score, current_score, score_kind):
            replacement_index = unique_docs.index(existing)
            unique_docs[replacement_index] = doc
            seen_by_doc_id[doc_id] = doc

    return unique_docs


def _build_context_from_docs(docs, stored_doc_map):
    context_parts = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id")
        stored_doc = stored_doc_map.get(doc_id, {})
        chunk_text = doc.metadata.get("chunk_text", doc.page_content)
        context_parts.append(
            "\n".join(
                [
                    f"Title: {stored_doc.get('title') or doc.metadata.get('title', 'Untitled')}",
                    f"Source type: {stored_doc.get('source_type') or doc.metadata.get('source_type', 'unknown')}",
                    f"Source value: {stored_doc.get('source_value') or doc.metadata.get('source_value', 'unknown')}",
                    f"Tags: {stored_doc.get('tags', '')}",
                    f"Saved at: {stored_doc.get('created_at', '')}",
                    f"Document date: {stored_doc.get('document_date', '')}",
                    f"Document summary: {stored_doc.get('summary', '')}",
                    f"Relevant excerpt:\n{chunk_text}",
                ]
            )
        )
    return "\n\n---\n\n".join(context_parts)


def _build_context_from_stored_documents(documents):
    context_parts = []
    for doc in documents:
        context_parts.append(
            "\n".join(
                [
                    f"Title: {doc.get('title', 'Untitled')}",
                    f"Source type: {doc.get('source_type', 'unknown')}",
                    f"Source value: {doc.get('source_value', 'unknown')}",
                    f"Tags: {doc.get('tags', '')}",
                    f"Saved at: {doc.get('created_at', '')}",
                    f"Document date: {doc.get('document_date', '')}",
                    f"Document summary: {doc.get('summary', '')}",
                    f"Content:\n{doc.get('content', '')}",
                ]
            )
        )
    return "\n\n---\n\n".join(context_parts)


def _build_sources_from_stored_documents(documents):
    sources = []
    for doc in documents:
        sources.append(
            {
                "title": doc.get("title", "Untitled"),
                "source_type": doc.get("source_type") or "document",
                "source_value": doc.get("source_value") or f"Document ID {doc.get('id')}",
                "tags": _normalize_tags(doc.get("tags", "")),
                "score": None,
                "score_kind": None,
                "created_at": doc.get("created_at"),
                "document_date": doc.get("document_date"),
                "summary": doc.get("summary"),
            }
        )
    return sources


def ask_question(q, tag_filter=None):
    llm = ChatOllama(model=OLLAMA_MODEL)
    candidate_docs = _select_candidate_documents(q, tag_filter)
    candidate_doc_ids = [doc["id"] for doc in candidate_docs]

    docs = []
    try:
        emb = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
        vs = FAISS.load_local(
            FAISS_INDEX_DIR,
            emb,
            allow_dangerous_deserialization=True,
        )
        retrieval_query = f"{q}\nCurrent local date: {datetime.now().strftime('%Y-%m-%d')}"
        docs = _search_documents(
            vs,
            retrieval_query,
            allowed_doc_ids=set(candidate_doc_ids) if candidate_doc_ids else None,
        )
        docs = _dedupe_documents(docs)[:TOP_K]
    except Exception:
        docs = []

    if docs:
        doc_ids = [doc.metadata.get("doc_id") for doc in docs if doc.metadata.get("doc_id") is not None]
        stored_docs = fetch_documents_by_ids(doc_ids)
        stored_doc_map = {doc["id"]: doc for doc in stored_docs}
        context = _build_context_from_docs(docs, stored_doc_map)
        answer = llm.invoke(
            "Answer the user's question using only the provided context. "
            "Use metadata like saved date, source type, and document summary when helpful.\n\n"
            f"Context:\n{context}\n\nQuestion: {q}"
        ).content

        sources = []
        for doc in docs:
            doc_id = doc.metadata.get("doc_id")
            stored_doc = stored_doc_map.get(doc_id, {})
            score = doc.metadata.get("_score")
            score_kind = doc.metadata.get("_score_kind")
            sources.append(
                {
                    "title": stored_doc.get("title") or doc.metadata.get("title", "Untitled"),
                    "source_type": stored_doc.get("source_type") or doc.metadata.get("source_type") or "document",
                    "source_value": stored_doc.get("source_value") or f"Document ID {doc_id}",
                    "tags": _normalize_tags(stored_doc.get("tags", doc.metadata.get("tags", []))),
                    "score": round(score, 4) if isinstance(score, (int, float)) else None,
                    "score_kind": score_kind,
                    "created_at": stored_doc.get("created_at"),
                    "document_date": stored_doc.get("document_date"),
                    "summary": stored_doc.get("summary"),
                }
            )
        return {"answer": answer, "sources": sources}

    if candidate_docs:
        context = _build_context_from_stored_documents(candidate_docs[:TOP_K])
        answer = llm.invoke(
            "Answer the user's question using only the provided context. "
            "If the question asks for a summary, summarize across all provided documents.\n\n"
            f"Context:\n{context}\n\nQuestion: {q}"
        ).content
        return {"answer": answer, "sources": _build_sources_from_stored_documents(candidate_docs[:TOP_K])}

    return {"answer": "No matching sources found for that question.", "sources": []}
