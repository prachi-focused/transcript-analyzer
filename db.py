"""
Database: transcript analyses + policy chunk embeddings (pgvector).

Configure via DATABASE_URL or POSTGRES_* (host, port, db, user, password).
Policy table needs `CREATE EXTENSION vector` once — see SETUP_DB.md.
"""
__all__ = [
    "store_transcript_analyses",
    "get_transcript_analyses",
    "list_transcript_ids_in_analyses",
    "transcript_analyses_is_empty",
    "policy_txt_chunks_is_empty",
    "store_policy_chunk_embeddings",
    "search_similar_policy_chunks",
]

# OpenAI text-embedding-3-small / ada-002
POLICY_EMBEDDING_DIMENSIONS = 1536
POLICY_SOURCE_DOCX = "policy_docx"
POLICY_SOURCE_TXT = "policy_txt"

import json
import os
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Config (not exported)
# -----------------------------------------------------------------------------

def _get_connection_params():
    if os.environ.get("DATABASE_URL"):
        return {"conninfo": os.environ["DATABASE_URL"]}
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "dbname": os.environ.get("POSTGRES_DB", "customer_support"),
        "user": os.environ.get("POSTGRES_USER", "prachi"),
        "password": os.environ.get("POSTGRES_PASSWORD", "postgres"),
    }


@contextmanager
def _connection():
    import psycopg2
    params = _get_connection_params()
    if "conninfo" in params:
        conn = psycopg2.connect(params["conninfo"])
    else:
        conn = psycopg2.connect(**params)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transcript_analyses (
                id SERIAL PRIMARY KEY,
                transcript_id TEXT NOT NULL UNIQUE,
                issues_identified JSONB NOT NULL DEFAULT '[]',
                issue_identification_time DOUBLE PRECISION NOT NULL,
                summary_of_issue TEXT NOT NULL,
                issue_identified_by_chatbot BOOLEAN NOT NULL,
                issue_identified_by_human_agent BOOLEAN NOT NULL,
                time_spent_with_chatbot_seconds DOUBLE PRECISION NOT NULL,
                time_spent_with_human_seconds DOUBLE PRECISION NOT NULL,
                time_spent_waiting_seconds DOUBLE PRECISION NOT NULL,
                resolution_stage TEXT NOT NULL,
                user_sentiment TEXT NOT NULL,
                stage_chatbot_failed TEXT NOT NULL,
                stage_human_failed TEXT NOT NULL,
                failure_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
                what_could_fix TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        # Ensure unique on transcript_id for upsert (idempotent for existing tables)
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS transcript_analyses_transcript_id_key "
            "ON transcript_analyses (transcript_id);"
        )


def _ensure_policy_chunks_table(conn):
    """
    Policy RAG chunks. Requires pgvector installed on the server and extension enabled.
    """
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            raise RuntimeError(
                "Enable pgvector on this database first:\n"
                "  brew install pgvector\n"
                "  psql -d customer_support -c 'CREATE EXTENSION vector;'\n"
                f"(Original error: {e})"
            ) from e
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS policy_chunks (
                id SERIAL PRIMARY KEY,
                embedding vector({POLICY_EMBEDDING_DIMENSIONS}) NOT NULL,
                source TEXT NOT NULL,
                document_name TEXT NOT NULL,
                section_heading TEXT NOT NULL DEFAULT '',
                chunk_index INT NOT NULL,
                uploaded_at TIMESTAMPTZ NOT NULL,
                content TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS policy_chunks_document_idx
            ON policy_chunks (document_name, chunk_index);
            """
        )


def _vector_param(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


# -----------------------------------------------------------------------------
# Public API: use these from any node (e.g. node_1_transcript_analysis or other files)
# -----------------------------------------------------------------------------


def list_transcript_ids_in_analyses() -> list[str]:
    """
    Return every ``transcript_id`` stored in ``transcript_analyses``, sorted for display.
    """
    with _connection() as conn:
        _ensure_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT transcript_id FROM transcript_analyses ORDER BY transcript_id"
            )
            return [row[0] for row in cur.fetchall()]


def transcript_analyses_is_empty() -> bool:
    """
    True if ``transcript_analyses`` has no rows.
    Ensures the table exists (a newly created table is treated as empty).
    """
    with _connection() as conn:
        _ensure_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT NOT EXISTS (SELECT 1 FROM transcript_analyses LIMIT 1)"
            )
            row = cur.fetchone()
            return bool(row[0]) if row else True


def policy_txt_chunks_is_empty() -> bool:
    """
    True if ``policy_chunks`` has no rows for TXT policy ingest (``source = policy_txt``).
    Matches TXT ingest in ``node_2_policy_update``. Ensures the table exists.
    """
    with _connection() as conn:
        _ensure_policy_chunks_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT NOT EXISTS (
                    SELECT 1 FROM policy_chunks WHERE source = %s LIMIT 1
                )
                """,
                (POLICY_SOURCE_TXT,),
            )
            row = cur.fetchone()
            return bool(row[0]) if row else True


def get_transcript_analyses(
    *,
    transcript_ids: list[str] | None = None,
    resolution_stage: str | list[str] | frozenset[str] | set[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[dict]:
    """
    Query stored transcript analyses from the DB. Optional filters.
    Returns list of dicts with new schema (time_spent_*, user_sentiment, stage_*_failed, etc.).

    ``resolution_stage`` may be a single stage string or a collection (e.g. resolved stages);
    collections are matched with ``= ANY(%s)``.
    """
    with _connection() as conn:
        _ensure_table(conn)
        with conn.cursor() as cur:
            conditions = []
            params = []
            if transcript_ids:
                conditions.append("transcript_id = ANY(%s)")
                params.append(transcript_ids)
            if resolution_stage is not None:
                if isinstance(resolution_stage, str):
                    conditions.append("resolution_stage = %s")
                    params.append(resolution_stage)
                else:
                    stages = list(resolution_stage)
                    conditions.append("resolution_stage = ANY(%s)")
                    params.append(stages)
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            limit_clause = f"LIMIT {int(limit)}" if limit is not None else ""
            cur.execute(
                f"""
                SELECT transcript_id, issues_identified, issue_identification_time,
                       summary_of_issue, issue_identified_by_chatbot,
                       issue_identified_by_human_agent,
                       time_spent_with_chatbot_seconds, time_spent_with_human_seconds,
                       time_spent_waiting_seconds, resolution_stage, user_sentiment,
                       stage_chatbot_failed, stage_human_failed,
                       failure_reasons, what_could_fix, created_at
                FROM transcript_analyses
                {where}
                ORDER BY created_at DESC
                {limit_clause}
                """,
                params if params else (),
            )
            columns = [d.name for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def store_transcript_analyses(analyses: list[dict]) -> None:
    """
    Upsert transcript analysis records: if transcript_id is not in the DB, insert;
    else update the existing row. Each item must match TranscriptAnalysis.model_dump()
    (see state.transcript_analysis_schema).
    """
    if not analyses:
        return
    with _connection() as conn:
        _ensure_table(conn)
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO transcript_analyses (
                    transcript_id, issues_identified, issue_identification_time,
                    summary_of_issue, issue_identified_by_chatbot,
                    issue_identified_by_human_agent,
                    time_spent_with_chatbot_seconds, time_spent_with_human_seconds,
                    time_spent_waiting_seconds, resolution_stage, user_sentiment,
                    stage_chatbot_failed, stage_human_failed,
                    failure_reasons, what_could_fix
                ) VALUES (
                    %(transcriptId)s, %(issuesIdentified)s::jsonb, %(issueIdentificationTime)s,
                    %(summary_of_issue)s, %(issueIdentifiedByChatbot)s,
                    %(issueIdentifiedByHumanAgent)s,
                    %(time_spent_with_chatbot_seconds)s, %(time_spent_with_human_seconds)s,
                    %(time_spent_waiting_seconds)s, %(resolution_stage)s, %(user_sentiment)s,
                    %(stage_chatbot_failed)s, %(stage_human_failed)s,
                    %(failure_reasons)s::jsonb, %(what_could_fix)s
                )
                ON CONFLICT (transcript_id) DO UPDATE SET
                    issues_identified = EXCLUDED.issues_identified,
                    issue_identification_time = EXCLUDED.issue_identification_time,
                    summary_of_issue = EXCLUDED.summary_of_issue,
                    issue_identified_by_chatbot = EXCLUDED.issue_identified_by_chatbot,
                    issue_identified_by_human_agent = EXCLUDED.issue_identified_by_human_agent,
                    time_spent_with_chatbot_seconds = EXCLUDED.time_spent_with_chatbot_seconds,
                    time_spent_with_human_seconds = EXCLUDED.time_spent_with_human_seconds,
                    time_spent_waiting_seconds = EXCLUDED.time_spent_waiting_seconds,
                    resolution_stage = EXCLUDED.resolution_stage,
                    user_sentiment = EXCLUDED.user_sentiment,
                    stage_chatbot_failed = EXCLUDED.stage_chatbot_failed,
                    stage_human_failed = EXCLUDED.stage_human_failed,
                    failure_reasons = EXCLUDED.failure_reasons,
                    what_could_fix = EXCLUDED.what_could_fix
                """,
                [
                    {
                        "transcriptId": a["transcriptId"],
                        "issuesIdentified": json.dumps(a.get("issuesIdentified", [])),
                        "issueIdentificationTime": float(a.get("issueIdentificationTime", 0)),
                        "summary_of_issue": a.get("summary_of_issue", ""),
                        "issueIdentifiedByChatbot": bool(a.get("issueIdentifiedByChatbot", False)),
                        "issueIdentifiedByHumanAgent": bool(a.get("issueIdentifiedByHumanAgent", False)),
                        "time_spent_with_chatbot_seconds": float(a.get("time_spent_with_chatbot_seconds", 0)),
                        "time_spent_with_human_seconds": float(a.get("time_spent_with_human_seconds", 0)),
                        "time_spent_waiting_seconds": float(a.get("time_spent_waiting_seconds", 0)),
                        "resolution_stage": a.get("resolution_stage", ""),
                        "user_sentiment": a.get("user_sentiment", "unknown"),
                        "stage_chatbot_failed": a.get("stage_chatbot_failed")
                        or a.get("point_chatbot_failed", "not_applicable"),
                        "stage_human_failed": a.get("stage_human_failed")
                        or a.get("point_human_failed", "not_applicable"),
                        "failure_reasons": json.dumps(
                            a.get("failure_reasons", []),
                            default=lambda o: o.value if hasattr(o, "value") else str(o),
                        ),
                        "what_could_fix": a.get("what_could_fix", "not_applicable"),
                    }
                    for a in analyses
                ],
            )


def store_policy_chunk_embeddings(
    rows: list[dict],
    embeddings: list[list[float]],
    *,
    source: str = POLICY_SOURCE_TXT,
) -> None:
    """
    Replace existing chunks for the same (source, document_name) then insert.
    Each row: content, document_name, section_heading, chunk_index, uploaded_at (datetime).
    """
    if not rows or len(rows) != len(embeddings):
        return
    if any(len(e) != POLICY_EMBEDDING_DIMENSIONS for e in embeddings):
        raise ValueError(
            f"Embeddings must have dimension {POLICY_EMBEDDING_DIMENSIONS} "
            "(use text-embedding-3-small or matching model)."
        )
    doc_names = list({r["document_name"] for r in rows})
    with _connection() as conn:
        _ensure_policy_chunks_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM policy_chunks WHERE source = %s AND document_name = ANY(%s)",
                (source, doc_names),
            )
            for row, emb in zip(rows, embeddings):
                cur.execute(
                    """
                    INSERT INTO policy_chunks (
                        embedding, source, document_name, section_heading,
                        chunk_index, uploaded_at, content
                    ) VALUES (%s::vector, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        _vector_param(emb),
                        source,
                        row["document_name"],
                        row.get("section_heading") or "",
                        int(row["chunk_index"]),
                        row["uploaded_at"],
                        row["content"],
                    ),
                )


def search_similar_policy_chunks(
    query_embedding: list[float],
    *,
    limit: int = 8,
    source: str | None = POLICY_SOURCE_TXT,
) -> list[dict]:
    """
    Cosine distance via pgvector (<=>). Returns chunks with metadata for citations.
    """
    if len(query_embedding) != POLICY_EMBEDDING_DIMENSIONS:
        raise ValueError(f"Query embedding dim must be {POLICY_EMBEDDING_DIMENSIONS}")
    with _connection() as conn:
        _ensure_policy_chunks_table(conn)
        with conn.cursor() as cur:
            if source:
                cur.execute(
                    """
                    SELECT document_name, section_heading, chunk_index, content,
                           uploaded_at, source,
                           (embedding <=> %s::vector) AS distance
                    FROM policy_chunks
                    WHERE source = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (_vector_param(query_embedding), source, _vector_param(query_embedding), limit),
                )
            else:
                cur.execute(
                    """
                    SELECT document_name, section_heading, chunk_index, content,
                           uploaded_at, source,
                           (embedding <=> %s::vector) AS distance
                    FROM policy_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (_vector_param(query_embedding), _vector_param(query_embedding), limit),
                )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
