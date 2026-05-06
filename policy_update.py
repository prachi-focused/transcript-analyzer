"""
Policy ingestion: load `.txt` from assets/policies → chunk → embed → store in Postgres (pgvector).
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

POLICIES_DIR = Path(__file__).resolve().parent / "assets" / "policies"
SOURCE_POLICY_TXT = "policy_txt"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 64


def load_policy_txt_paths(policies_dir: Path | None = None) -> list[Path]:
    """All `.txt` files under `assets/policies/` (excludes temp `~$` files)."""
    root = policies_dir or POLICIES_DIR
    if not root.is_dir():
        return []
    paths = sorted(
        p for p in root.glob("*.txt") if p.is_file() and not p.name.startswith("~$")
    )
    return paths


def parse_txt_into_sections(doc_path: Path) -> tuple[str, list[tuple[str, str]]]:
    """
    Read one `.txt` into (document_name, [(section_heading, body_text), ...]).
    Sections split on markdown-style headings: lines starting with # / ## / ### …
    If no headings, the whole file is one section.
    """
    raw = doc_path.read_text(encoding="utf-8", errors="replace")
    document_name = doc_path.name
    body = raw.strip()
    if not body:
        return document_name, []

    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(body))
    if not matches:
        return document_name, [("(whole document)", body)]

    sections: list[tuple[str, str]] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(("(preamble)", preamble))
    for i, m in enumerate(matches):
        heading = m.group(2).strip() or f"section_{i}"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[start:end].strip()
        if chunk:
            sections.append((heading, chunk))
    if not sections:
        return document_name, [("(whole document)", body)]
    return document_name, sections


def load_all_policy_documents(paths: list[Path]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Step 1 output: list of (document_name, sections) for each `.txt` file."""
    out: list[tuple[str, list[tuple[str, str]]]] = []
    for path in paths:
        name, sections = parse_txt_into_sections(path)
        if sections:
            out.append((name, sections))
    return out


def split_sections_into_chunks(
    document_name: str,
    sections: list[tuple[str, str]],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    uploaded_at: datetime | None = None,
) -> list[dict]:
    """
    Step 2: split each section body into overlapping chunks.
    Each row dict: content, document_name, section_heading, chunk_index, uploaded_at.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    uploaded = uploaded_at or datetime.now(timezone.utc)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    rows: list[dict] = []
    chunk_index = 0
    for section_heading, body in sections:
        if not body.strip():
            continue
        for chunk_text in splitter.split_text(body):
            rows.append(
                {
                    "content": chunk_text,
                    "document_name": document_name,
                    "section_heading": section_heading,
                    "chunk_index": chunk_index,
                    "uploaded_at": uploaded,
                }
            )
            chunk_index += 1
    return rows


def chunk_all_documents(
    loaded: list[tuple[str, list[tuple[str, str]]]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Run split for every document; single `uploaded_at` per ingest run."""
    uploaded_at = datetime.now(timezone.utc)
    all_rows: list[dict] = []
    for document_name, sections in loaded:
        all_rows.extend(
            split_sections_into_chunks(
                document_name,
                sections,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                uploaded_at=uploaded_at,
            )
        )
    return all_rows


def embed_policy_chunks(
    rows: list[dict],
    *,
    model: str | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Step 3: OpenAI embeddings for each chunk `content` (batched)."""
    from langchain_openai import OpenAIEmbeddings

    model = model or os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
    embedder = OpenAIEmbeddings(model=model)
    texts = [r["content"] for r in rows]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors.extend(embedder.embed_documents(batch))
    return vectors


def persist_policy_embeddings(rows: list[dict], embeddings: list[list[float]]) -> int:
    """
    Step 4: write to Postgres with source=policy_txt and citation metadata.
    Replaces prior rows for the same document_name + source.
    """
    from db import store_policy_chunk_embeddings

    if not rows:
        return 0
    store_policy_chunk_embeddings(rows, embeddings, source=SOURCE_POLICY_TXT)
    return len(rows)


def run_policy_ingest_pipeline(
    policies_dir: Path | None = None,
) -> dict:
    """
    Run load → chunk → embed → store in order.
    Returns summary dict (counts, paths). No-op if no `.txt` files.
    """
    paths = load_policy_txt_paths(policies_dir)
    if not paths:
        return {
            "ok": False,
            "message": f"No .txt files in {POLICIES_DIR}. Add policy text files.",
            "documents": 0,
            "chunks": 0,
        }

    loaded = load_all_policy_documents(paths)
    rows = chunk_all_documents(loaded, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    if not rows:
        return {
            "ok": False,
            "message": "Documents parsed but produced no chunks.",
            "documents": len(loaded),
            "chunks": 0,
        }

    embeddings = embed_policy_chunks(rows)
    n = persist_policy_embeddings(rows, embeddings)
    return {
        "ok": True,
        "message": f"Ingested {n} chunks from {len(loaded)} document(s).",
        "documents": len(loaded),
        "chunks": n,
        "files": [p.name for p in paths],
    }


def search_policy_context(query: str, *, k: int = 6) -> list[dict]:
    """Embed query and return top chunks (for Q&A / citations)."""
    from langchain_openai import OpenAIEmbeddings
    from db import search_similar_policy_chunks

    model = os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)
    embedder = OpenAIEmbeddings(model=model)
    qvec = embedder.embed_query(query)
    return search_similar_policy_chunks(qvec, limit=k, source=SOURCE_POLICY_TXT)


def policy_update(state: dict) -> dict:
    """LangGraph node: full policy ingest pipeline."""
    print("Running policy store update (TXT → chunks → embeddings → pgvector)...")
    result = run_policy_ingest_pipeline()
    print(result.get("message", result))
    return {}
