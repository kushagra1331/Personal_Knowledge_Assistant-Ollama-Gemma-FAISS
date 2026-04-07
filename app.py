import os
import tempfile
import time
import streamlit as st

from db import init_db, fetch_documents
from extractors import (
    extract_text_from_url,
    extract_text_from_pdf,
    extract_text_from_raw_text,
)
from ingest import ingest_documents
from rag import ask_question
from utils.helpers import build_documents, normalize_tags

init_db()

st.set_page_config(page_title="Knowledge Copilot", layout="wide")
st.markdown(
    """
    <style>
    @keyframes thinking-blink {
        0% { opacity: 1; }
        50% { opacity: 0.35; }
        100% { opacity: 1; }
    }
    .thinking-indicator {
        animation: thinking-blink 1s ease-in-out infinite;
        font-weight: 600;
        color: #b45309;
        padding: 0.75rem 0.9rem;
        border-radius: 0.6rem;
        background: #fff7ed;
        border: 1px solid #fdba74;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🧠 Personal Knowledge Assistant")

tab_ingest, tab_ask, tab_browse = st.tabs(["Add Knowledge", "Ask Questions", "Browse Saved Items"])

with tab_ingest:
    st.subheader("Add Knowledge")

    input_type = st.radio("Choose input type", ["URL", "Raw Text", "PDF"], horizontal=True)
    tags_input = st.text_input(
        "Tags (comma-separated)",
        placeholder="rag, local-llm, productivity",
    )
    tags = normalize_tags(tags_input)

    if input_type == "URL":
        url = st.text_input("Article URL")

        if st.button("Ingest URL"):
            if not url:
                st.warning("Please enter a URL.")
            else:
                try:
                    with st.status("Ingesting URL...", expanded=True) as status:
                        st.write("Fetching content from the URL...")
                        title, text = extract_text_from_url(url)

                        if not text.strip():
                            status.update(label="URL ingestion failed", state="error")
                            st.error("No usable text could be extracted from the URL.")
                        else:
                            st.write("Preparing document for storage...")
                            docs = build_documents(
                                title,
                                text,
                                tags,
                                source_type="url",
                                source_value=url,
                            )
                            st.write("Creating embeddings and saving to the knowledge base...")
                            ingest_documents(docs)
                            status.update(label="URL ingested", state="complete")
                            st.success(f"Ingested '{title}'.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Failed to ingest URL: {e}")

    elif input_type == "Raw Text":
        title = st.text_input("Title")
        raw_text = st.text_area("Paste text here", height=220)

        if st.button("Ingest Text"):
            if not raw_text.strip():
                st.warning("Please paste some text.")
            else:
                try:
                    with st.status("Ingesting text...", expanded=True) as status:
                        st.write("Preparing text content...")
                        final_title, text = extract_text_from_raw_text(title, raw_text)
                        st.write("Creating document record...")
                        docs = build_documents(
                            final_title,
                            text,
                            tags,
                            source_type="raw_text",
                            source_value=final_title,
                        )
                        st.write("Creating embeddings and saving to the knowledge base...")
                        ingest_documents(docs)
                        status.update(label="Text ingested", state="complete")
                        st.success(f"Ingested '{final_title}'.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to ingest text: {e}")

    else:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        if st.button("Ingest PDF"):
            if not uploaded_file:
                st.warning("Please upload a PDF.")
            else:
                temp_path = None
                try:
                    with st.status("Ingesting PDF...", expanded=True) as status:
                        st.write("Uploading PDF for processing...")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            temp_path = tmp.name

                        st.write("Extracting text from the PDF...")
                        extracted_title, text = extract_text_from_pdf(temp_path)
                        final_title = uploaded_file.name or extracted_title

                        if not text.strip():
                            status.update(label="PDF ingestion failed", state="error")
                            st.error("No usable text could be extracted from the PDF.")
                        else:
                            st.write("Preparing document for storage...")
                            docs = build_documents(
                                final_title,
                                text,
                                tags,
                                source_type="pdf",
                                source_value=uploaded_file.name or temp_path,
                            )
                            st.write("Creating embeddings and saving to the knowledge base...")
                            ingest_documents(docs)
                            status.update(label="PDF ingested", state="complete")
                            st.success(f"Ingested '{final_title}'.")
                            st.rerun()
                except Exception as e:
                    st.error(f"Failed to ingest PDF: {e}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)

with tab_ask:
    st.subheader("Ask Questions")
    question = st.text_area("Your question", height=120)
    tag_filter = st.text_input("Optional tag filter for question search", placeholder="rag")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            thinking_placeholder = st.empty()
            try:
                start_time = time.perf_counter()
                thinking_placeholder.markdown(
                    "<div class='thinking-indicator'>Thinking...</div>",
                    unsafe_allow_html=True,
                )
                response = ask_question(question, tag_filter or None)

                elapsed = time.perf_counter() - start_time
                st.markdown("### Answer")
                st.caption(f"Response time: {elapsed:.2f} seconds")
                st.write(response["answer"])

                st.markdown("### Sources")
                if response["sources"]:
                    for src in response["sources"]:
                        tags_display = ", ".join(src["tags"]) if src["tags"] else ""
                        created_display = (
                            f" | saved: {src['created_at']}"
                            if src.get("created_at")
                            else ""
                        )
                        document_date_display = (
                            f" | document date: {src['document_date']}"
                            if src.get("document_date")
                            else ""
                        )
                        score_display = ""
                        if src.get("score") is not None:
                            if src.get("score_kind") == "distance":
                                score_display = f" | distance: {src['score']}"
                            else:
                                score_display = f" | match score: {src['score']}"
                        st.markdown(
                            f"- **{src['title']}** | `{src['source_type']}` | "
                            f"{src['source_value']} | tags: {tags_display}"
                            f"{created_display}{document_date_display}{score_display}"
                        )
                else:
                    st.info("No sources matched.")
            except Exception as e:
                st.error(f"Failed to answer question: {e}")
            finally:
                thinking_placeholder.empty()

with tab_browse:
    st.subheader("Browse Saved Items")

    all_items = fetch_documents(limit=1000)
    all_tags = set()
    for item in all_items:
        raw_tags = item.get("tags", "") or ""
        for tag in raw_tags.split(","):
            cleaned = tag.strip()
            if cleaned:
                all_tags.add(cleaned)

    if all_tags:
        st.markdown("**Available tags**")
        st.write(", ".join(sorted(all_tags)))
    else:
        st.info("No tags saved yet.")

    browse_tag = st.text_input("Browse tag", placeholder="rag")
    items = fetch_documents(tag=browse_tag or None, limit=100)

    if not items:
        st.info("No saved items found.")
    else:
        for item in items:
            with st.expander(f"{item['title']} (ID: {item['id']})"):
                if item.get("source_type"):
                    st.write(f"**Source type:** {item['source_type']}")
                if item.get("source_value"):
                    st.write(f"**Source value:** {item['source_value']}")
                st.write(f"**Tags:** {item['tags']}")
                if item.get("created_at"):
                    st.write(f"**Saved:** {item['created_at']}")
                if item.get("document_date"):
                    st.write(f"**Document date:** {item['document_date']}")
                if item.get("summary"):
                    st.write(f"**Summary:** {item['summary']}")
                preview = item["content"][:1000] + ("..." if len(item["content"]) > 1000 else "")
                st.write("**Preview:**")
                st.write(preview)
