import hashlib
import re

from langchain_core.documents import Document


def normalize_tags(tag_string):
    return [t.strip() for t in tag_string.split(",") if t.strip()]


def build_summary(text, max_chars=320):
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def extract_document_date(text):
    patterns = [
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
        r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "")
        if match:
            return match.group(1)
    return None


def build_content_hash(title, text, source_type=None, source_value=None):
    digest = hashlib.sha256()
    payload = "||".join(
        [
            title or "",
            text or "",
            source_type or "",
            source_value or "",
        ]
    )
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def build_documents(title, text, tags, source_type, source_value):
    summary = build_summary(text)
    document_date = extract_document_date(text)
    content_hash = build_content_hash(title, text, source_type=source_type, source_value=source_value)
    return [
        Document(
            page_content=text,
            metadata={
                "title": title,
                "tags": tags,
                "source_type": source_type,
                "source_value": source_value,
                "summary": summary,
                "document_date": document_date,
                "content_hash": content_hash,
            },
        )
    ]
