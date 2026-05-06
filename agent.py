"""LangGraph workflow: route transcript source, then analyze or load from DB."""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from uuid import uuid4

from analyze_transcript import analyze_transcript
from load_DB_analysis import load_DB_analysis
from router import transcript_source_router
from state.workflow_state import CustomerSupportProcess

load_dotenv()


def check_transcript_source(state: dict) -> dict:
    decision = transcript_source_router(state)
    if decision == "analyze_transcript":
        return {"transcript_source_choice": "run_new_analysis"}
    return {"transcript_source_choice": "load_from_db"}


def route_after_transcript_source_check(state: dict) -> str:
    return str(state.get("transcript_source_choice", "run_new_analysis"))


def build_graph():
    builder = StateGraph(CustomerSupportProcess)
    builder.add_node("check_transcript_source", check_transcript_source)
    builder.add_node("load_DB_analysis", load_DB_analysis)
    builder.add_node("analyze_transcript", analyze_transcript)

    builder.add_edge(START, "check_transcript_source")
    builder.add_conditional_edges(
        "check_transcript_source",
        route_after_transcript_source_check,
        {
            "run_new_analysis": "analyze_transcript",
            "load_from_db": "load_DB_analysis",
        },
    )

    builder.add_edge("analyze_transcript", END)
    builder.add_edge("load_DB_analysis", END)
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
