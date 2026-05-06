"""LangGraph workflow: transcript source → analyze or load from DB → optional policy ingest."""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from uuid import uuid4

from analyze_transcript import analyze_transcript
from load_DB_analysis import load_DB_analysis
from policy_update import policy_update
from router import policy_update_router, transcript_source_router
from state.workflow_state import CustomerSupportProcess

load_dotenv()


def check_transcript_source(state: dict) -> dict:
    decision = transcript_source_router(state)
    if decision == "analyze_transcript":
        return {"transcript_source_choice": "run_new_analysis"}
    return {"transcript_source_choice": "load_from_db"}


def route_after_transcript_source_check(state: dict) -> str:
    return str(state.get("transcript_source_choice", "run_new_analysis"))


def route_policy_update(state: dict) -> dict:
    decision = policy_update_router(state)
    if decision == "policy_update":
        return {"policy_update_choice": "run_policy_update"}
    return {"policy_update_choice": "skip_policy_update"}


def route_after_policy(state: dict) -> str:
    return str(state.get("policy_update_choice", "run_policy_update"))


def build_graph():
    builder = StateGraph(CustomerSupportProcess)
    builder.add_node("check_transcript_source", check_transcript_source)
    builder.add_node("load_DB_analysis", load_DB_analysis)
    builder.add_node("analyze_transcript", analyze_transcript)
    builder.add_node("route_policy_update", route_policy_update)
    builder.add_node("policy_update", policy_update)

    builder.add_edge(START, "check_transcript_source")
    builder.add_conditional_edges(
        "check_transcript_source",
        route_after_transcript_source_check,
        {
            "run_new_analysis": "analyze_transcript",
            "load_from_db": "load_DB_analysis",
        },
    )

    builder.add_edge("analyze_transcript", "route_policy_update")
    builder.add_edge("load_DB_analysis", "route_policy_update")
    builder.add_conditional_edges(
        "route_policy_update",
        route_after_policy,
        {
            "run_policy_update": "policy_update",
            "skip_policy_update": END,
        },
    )
    builder.add_edge("policy_update", END)
    return builder.compile()


graph = build_graph()


if __name__ == "__main__":
    message = [HumanMessage(content="Analyze the transcript and return the analysis.", role="user")]
    transcripts = [
        "transcript_01",
        "transcript_02",
        "transcript_03",
        "transcript_04",
        "transcript_05",
        "transcript_06",
        "transcript_07",
        "transcript_08",
        "transcript_09",
        "transcript_10",
        "transcript_11",
        "transcript_12",
        "transcript_13",
        "transcript_14",
    ]
    response = graph.invoke(
        {"messages": message, "transcripts": transcripts, "path_to_transcripts": "assets/transcripts/"},
        config={"configurable": {"thread_id": f"run-{uuid4()}"}},
    )
