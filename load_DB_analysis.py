"""Load existing transcript analyses from Postgres into graph state."""

from __future__ import annotations

import json
from typing import Any

from db import get_transcript_analyses
from state.transcript_analysis_schema import TranscriptAnalysis


def _row_to_transcript_analysis(row: dict[str, Any]) -> TranscriptAnalysis:
    issues = row["issues_identified"]
    if isinstance(issues, str):
        issues = json.loads(issues)
    fr = row.get("failure_reasons") or []
    if isinstance(fr, str):
        fr = json.loads(fr)
    payload = {
        "transcriptId": row["transcript_id"],
        "issuesIdentified": issues,
        "issueIdentificationTime": float(row["issue_identification_time"]),
        "summary_of_issue": row["summary_of_issue"],
        "issueIdentifiedByChatbot": bool(row["issue_identified_by_chatbot"]),
        "issueIdentifiedByHumanAgent": bool(row["issue_identified_by_human_agent"]),
        "time_spent_with_chatbot_seconds": float(row["time_spent_with_chatbot_seconds"]),
        "time_spent_with_human_seconds": float(row["time_spent_with_human_seconds"]),
        "time_spent_waiting_seconds": float(row["time_spent_waiting_seconds"]),
        "resolution_stage": row["resolution_stage"],
        "user_sentiment": row["user_sentiment"],
        "stage_chatbot_failed": row["stage_chatbot_failed"],
        "stage_human_failed": row["stage_human_failed"],
        "failure_reasons": fr,
        "what_could_fix": row["what_could_fix"],
    }
    return TranscriptAnalysis.model_validate(payload)


def load_DB_analysis(state: dict) -> dict:
    """
    Fetch analyses for ``state['transcripts']`` from ``transcript_analyses``, ordered to match that list.
    """
    print("Loading transcript analyses from database...")
    ids = state.get("transcripts") or []
    if not ids:
        print("No transcripts in state; nothing to load from DB.")
        return {"transcript_analysis": []}

    try:
        rows = get_transcript_analyses(transcript_ids=ids)
    except Exception as e:
        print(f"Could not load from database: {e}")
        return {"transcript_analysis": [], "error": str(e)}

    by_id = {r["transcript_id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    for m in missing:
        print(f"Transcript not found in DB: {m}")

    ordered_rows = [by_id[i] for i in ids if i in by_id]
    analyses = [_row_to_transcript_analysis(r) for r in ordered_rows]
    print(f"Loaded {len(analyses)} transcript analysis record(s) from DB.")
    return {"transcript_analysis": analyses}
