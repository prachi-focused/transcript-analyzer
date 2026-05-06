from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Short enums for structured answers (no long text except summary)
ResolutionStage = Literal[
    "chatbot_resolved",
    "human_resolved",
    "unresolved",
    "escalated_no_resolution",
    "user_abandoned",
    "transferred_only",
    "partial_resolution",
    "refund_processed",
    "unknown",
]
UserSentiment = Literal[
    "happy",
    "satisfied",
    "neutral",
    "frustrated",
    "irritated",
    "sad",
    "angry",
    "unknown",
]
PointFailed = Literal[
    "not_applicable",
    "could_not_identify",
    "policy_denied",
    "wrong_info",
    "could_not_process",
    "suggested_handoff_to_human",
    "timeout_waiting_for_user_response",
    "timeout_waiting_for_support_agent_response",
    "timeout_waiting_for_chatbot_response",
    "user_left_conversation",
    "other",
]
FailureCategory = Literal[
    "not_applicable",
    "policy",
    "system_tooling",
    "knowledge_capability",
    "agent_communication",
    "chatbot_communication",
    "user_behavior",
    "chat_abandoned",
    "other",
]

# Single source of truth: reason string -> FailureCategory value
REASON_TO_CATEGORY: dict[str, FailureCategory] = {
    # Policy
    "policy_violation": "policy",
    "policy_does_not_allow_request": "policy",
    "refund_window_expired": "policy",
    "eligibility_criteria_not_met": "policy",
    # System / tooling
    "system_error": "system_tooling",
    "payment_system_failure": "system_tooling",
    "booking_system_error": "system_tooling",
    "chatbot_or_integration_down": "system_tooling",
    "session_or_timeout_error": "system_tooling",
    "agent_missing_tool_or_integration": "system_tooling",
    "chatbot_missing_tool_or_integration": "system_tooling",
    "agent_tool_or_system_access_limited": "system_tooling",
    "agent_could_not_access_user_data": "system_tooling",
    "chatbot_could_not_access_user_data": "system_tooling",
    # Knowledge / capability
    "agent_inadequate_knowledge": "knowledge_capability",
    "chatbot_inadequate_knowledge": "knowledge_capability",
    "agent_inadequate_capabilities": "knowledge_capability",
    "chatbot_inadequate_capabilities": "knowledge_capability",
    "agent_lacked_authority_to_resolve": "knowledge_capability",
    "agent_did_not_follow_process": "knowledge_capability",
    "agent_skipped_verification_step": "knowledge_capability",
    # Communication (split by party)
    "agent_gave_incorrect_or_confusing_instructions": "agent_communication",
    "agent_misunderstood_user_request": "agent_communication",
    "agent_used_jargon_or_unclear_language": "agent_communication",
    "agent_failed_to_confirm_understanding": "agent_communication",
    "chatbot_interpreted_request_incorrectly": "chatbot_communication",
    "chatbot_gave_ambiguous_or_wrong_response": "chatbot_communication",
    "chatbot_repeated_unhelpful_answer": "chatbot_communication",
    "chatbot_did_not_acknowledge_user_concern": "chatbot_communication",
    # User behavior / input
    "user_misunderstanding_policy": "user_behavior",
    "user_provided_invalid_information": "user_behavior",
    "user_provided_incomplete_information": "user_behavior",
    "user_provided_wrong_account_or_order_details": "user_behavior",
    "user_request_outside_scope": "user_behavior",
    "user_request_was_ambiguous": "user_behavior",
    "user_changed_or_clarified_request_mid_flow": "user_behavior",
    "user_did_not_provide_required_details": "user_behavior",
    "user_expectations_mismatch": "user_behavior",
    # Abandonment
    "user_left_conversation": "chat_abandoned",
    "user_abandoned_mid_flow": "chat_abandoned",
    "user_closed_chat_without_resolution": "chat_abandoned",
    # Default / N/A
    "other": "other",
    "not_applicable": "not_applicable",
}

FailureReason = StrEnum(
    "FailureReason",
    [(k, k) for k in REASON_TO_CATEGORY],
)


class FailureReasonItem(BaseModel):
    """One failure reason plus category; category is aligned to ``REASON_TO_CATEGORY``."""

    reason: FailureReason
    category: FailureCategory

    @model_validator(mode="after")
    def align_category_to_reason(self) -> "FailureReasonItem":
        expected: FailureCategory = REASON_TO_CATEGORY[self.reason.value]
        if self.category != expected:
            return self.model_copy(update={"category": expected})
        return self


WhatCouldFix = Literal[
    "not_applicable",
    "policy_change",
    "chatbot_training",
    "faster_handoff",
    "clearer_messaging",
    "tool_access",
    "other",
]

# Stages counted as resolved for aggregate metrics (subset of ResolutionStage values).
RESOLVED_RESOLUTION_STAGES: frozenset[str] = frozenset(
    {"chatbot_resolved", "human_resolved", "refund_processed"}
)


class TranscriptAnalysis(BaseModel):
    transcriptId: str
    issuesIdentified: list[str]  # short labels, e.g. ["refund", "email not received"]

    issueIdentificationTime: float  # seconds to identify issue
    summary_of_issue: str  # only lengthy field: what happened and outcome

    issueIdentifiedByChatbot: bool
    issueIdentifiedByHumanAgent: bool

    time_spent_with_chatbot_seconds: float
    time_spent_with_human_seconds: float
    time_spent_waiting_seconds: float

    resolution_stage: ResolutionStage

    user_sentiment: UserSentiment  # from tone of user messages

    stage_chatbot_failed: PointFailed  # at what point chatbot could not resolve (or not_applicable)
    stage_human_failed: PointFailed  # at what point human could not resolve (or not_applicable)

    # [] only when resolution_stage is in RESOLVED_RESOLUTION_STAGES; otherwise 1–3 items
    failure_reasons: list[FailureReasonItem] = Field(default_factory=list, max_length=3)

    what_could_fix: WhatCouldFix

    @model_validator(mode="after")
    def failure_reasons_consistent_with_resolution(self) -> "TranscriptAnalysis":
        if self.resolution_stage in RESOLVED_RESOLUTION_STAGES:
            if self.failure_reasons:
                return self.model_copy(update={"failure_reasons": []})
            return self
        if not self.failure_reasons:
            return self.model_copy(
                update={
                    "failure_reasons": [
                        FailureReasonItem(
                            reason=FailureReason.other,
                            category=REASON_TO_CATEGORY[FailureReason.other.value],
                        )
                    ]
                }
            )
        return self
