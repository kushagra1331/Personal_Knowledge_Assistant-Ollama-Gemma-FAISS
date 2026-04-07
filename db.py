import os
import sqlite3

from config import SQLITE_DB_PATH


DOCUMENT_COLUMNS = [
    "id",
    "title",
    "content",
    "tags",
    "created_at",
    "source_type",
    "source_value",
    "summary",
    "document_date",
    "content_hash",
]


def _connect():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _existing_columns(conn):
    return {row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}


def init_db():
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            tags TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    columns = _existing_columns(conn)
    missing_columns = {
        "source_type": "ALTER TABLE documents ADD COLUMN source_type TEXT",
        "source_value": "ALTER TABLE documents ADD COLUMN source_value TEXT",
        "summary": "ALTER TABLE documents ADD COLUMN summary TEXT",
        "document_date": "ALTER TABLE documents ADD COLUMN document_date TEXT",
        "content_hash": "ALTER TABLE documents ADD COLUMN content_hash TEXT",
        "created_at": "ALTER TABLE documents ADD COLUMN created_at TEXT",
    }
    for column_name, statement in missing_columns.items():
        if column_name not in columns:
            conn.execute(statement)

    conn.execute(
        "UPDATE documents SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
    )
    conn.commit()
    conn.close()


def insert_document(
    title,
    content,
    tags,
    source_type=None,
    source_value=None,
    summary=None,
    document_date=None,
    content_hash=None,
):
    conn = _connect()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (
            title,
            content,
            tags,
            created_at,
            source_type,
            source_value,
            summary,
            document_date,
            content_hash
        ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
        """,
        (
            title,
            content,
            ",".join(tags),
            source_type,
            source_value,
            summary,
            document_date,
            content_hash,
        ),
    )
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def _build_document_query(tag=None, created_on=None, source_type=None, limit=100):
    query = "SELECT {} FROM documents"
    conditions = []
    params = []

    if tag:
        conditions.append("tags LIKE ?")
        params.append(f"%{tag}%")

    if created_on:
        conditions.append("date(created_at, 'localtime') = ?")
        params.append(created_on)

    if source_type:
        conditions.append("source_type = ?")
        params.append(source_type)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return query, params


def fetch_documents(tag=None, limit=100, created_on=None, source_type=None):
    conn = _connect()
    query, params = _build_document_query(
        tag=tag,
        created_on=created_on,
        source_type=source_type,
        limit=limit,
    )
    rows = conn.execute(query.format(", ".join(DOCUMENT_COLUMNS)), tuple(params)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_documents_by_ids(doc_ids):
    if not doc_ids:
        return []

    conn = _connect()
    placeholders = ",".join(["?"] * len(doc_ids))
    rows = conn.execute(
        f"""
        SELECT {", ".join(DOCUMENT_COLUMNS)}
        FROM documents
        WHERE id IN ({placeholders})
        """,
        tuple(doc_ids),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_document_by_hash(content_hash):
    if not content_hash:
        return None

    conn = _connect()
    row = conn.execute(
        f"""
        SELECT {", ".join(DOCUMENT_COLUMNS)}
        FROM documents
        WHERE content_hash = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (content_hash,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None
