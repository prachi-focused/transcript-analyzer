from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from db import store_transcript_analyses
from state.transcript_analysis_schema import TranscriptAnalysis

load_dotenv()

TRANSCRIPT_ANALYSIS_SYSTEM_MESSAGE = """
    You are an internal support operations analyst for a movie theater chain.
    Analyze the transcript and fill the structured fields. Use only the allowed enum values and numbers; keep text short except summary_of_issue.

    Time (in seconds, estimate from message flow if not explicit):
    - time_spent_with_chatbot_seconds: time user spent interacting with the chatbot only.
    - time_spent_with_human_seconds: time user spent with a human agent only.
    - time_spent_waiting_seconds: time waiting for reply or in queue.

    Resolution:
    - resolution_stage: how the chat ended (ResolutionStage enum)

    User tone (from wording and cues):
    - user_sentiment: use UserSentiment enum.

    Where resolution broke down (use not_applicable if that party resolved it):
    - stage_chatbot_failed: PointFailed — where the chatbot failed, or not_applicable.
    - stage_human_failed: PointFailed — where the human failed, or not_applicable if no human agent was involved or human resolved.

    Failure reasons (must agree with resolution_stage):
    - failure_reasons: each item has `reason` and `category` (see REASON_TO_CATEGORY).
    - If resolution_stage is chatbot_resolved, human_resolved, or refund_processed: failure_reasons MUST be exactly [] (no exceptions — the issue was resolved; do not record root-cause reasons here).
    - For any other resolution_stage : failure_reasons MUST have 1 to 2 items — the main causes that explain why it was not a clean resolve, most important first.
    - Each `reason` MUST be one of the predefined failure reason strings from the schema (the same set used in REASON_TO_CATEGORY).
    - Each `category` MUST match REASON_TO_CATEGORY for that reason (e.g. user_expectations_mismatch → user_behavior; agent_* messaging issues → agent_communication; chatbot_* messaging issues → chatbot_communication; policy → policy; abandonment → chat_abandoned).
    - Do not repeat the same `reason` twice in the list.

    What could improve outcomes:
    - what_could_fix: WhatCouldFix; use not_applicable when the interaction was fully resolved.

    Other:
    - issuesIdentified: short list of issue labels (use IssueIdentified enum).
    - summary_of_issue: 2–4 sentences on what happened and the outcome (only field that can be longer).
    - issueIdentifiedByChatbot / issueIdentifiedByHumanAgent: true only if that agent clearly identified and stated the issue.
    Be strict: only use enum values defined in the response schema; estimate times from turn-taking when not stated.
    """

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_model = model.with_structured_output(TranscriptAnalysis)


def analyze_single_transcript(transcript_id: str, transcripts_dir: Path) -> TranscriptAnalysis:
    system_message = TRANSCRIPT_ANALYSIS_SYSTEM_MESSAGE
    path_to_transcript = transcripts_dir / f"{transcript_id}.txt"
    with open(path_to_transcript, "r") as file:
        transcript_content = file.read()
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=f"Transcript id: {transcript_id}\n\n{transcript_content}"),
    ]
    return structured_model.invoke(messages)


MAX_CONCURRENCY = 5


def analyze_transcript(state: dict) -> dict:
    """
    LangGraph node: runs transcript analysis (up to MAX_CONCURRENCY in parallel),
    stores results in the local Postgres DB when available, and returns a state update.
    """
    print("Running transcript analysis node...")
    transcript_ids = state.get("transcripts", ["transcript_01"])
    raw = state.get("path_to_transcripts", "assets/transcripts/")
    transcripts_dir = Path(raw).expanduser().resolve()

    found_ids: list[str] = []
    for tid in transcript_ids:
        candidate = transcripts_dir / f"{tid}.txt"
        if not candidate.is_file():
            print(f"Transcript not found: {tid}")
            continue
        found_ids.append(tid)

    if not found_ids:
        print("No transcript files found; skipping analysis.")
        return {"transcript_analysis": []}

    def _run(tid: str) -> TranscriptAnalysis:
        return analyze_single_transcript(tid, transcripts_dir)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        analysis = list(executor.map(_run, found_ids))
    try:
        store_transcript_analyses([a.model_dump(mode="json") for a in analysis])
    except Exception as e:
        import warnings

        warnings.warn(f"Could not store analyses in DB (skipping): {e}", stacklevel=0)
    return {"transcript_analysis": analysis}
