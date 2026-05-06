from __future__ import annotations

from typing import NotRequired

from langgraph.graph.message import MessagesState

from .transcript_analysis_schema import TranscriptAnalysis


class CustomerSupportProcess(MessagesState):
    """LangGraph state for the customer support workflow."""

    transcript_analysis: NotRequired[list[TranscriptAnalysis]]
    transcripts: list[str] = []
    path_to_transcripts: str = "assets/transcripts/"
    transcript_source_choice: str = ""
    policy_update_choice: str = ""
    operations_metrics: dict = {}
    failure_metrics: dict = {}
    error: str = ""
# too many things in state, need to clean up