"""Minimal LangGraph-style agent: linear graph built via build_graph()."""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

load_dotenv()

_TEXT_AGENT = create_agent(
    ChatOpenAI(model="gpt-4o-mini"),
    tools=[],
    system_prompt=(
        "You are a helpful assistant that reads the user's text and responds "
        "clearly and concisely."
    ),
)

class AgentState(TypedDict, total=False):
    """State flowing through the graph."""

    input_text: str
    normalized: str
    output: str


def normalize(state: AgentState) -> dict[str, str]:
    text = (state.get("input_text") or "").strip()
    return {"normalized": text}


def respond(state: AgentState) -> dict[str, str]:
    body = state.get("normalized", "")
    if not body:
        return {"output": "(empty input)"}
    result = _TEXT_AGENT.invoke(
        {"messages": [HumanMessage(content=body)]},
    )
    last = result["messages"][-1]
    out = last.content
    if not isinstance(out, str):
        out = str(out)
    return {"output": out}


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("normalize", normalize)
    workflow.add_node("respond", respond)
    workflow.add_edge(START, "normalize")
    workflow.add_edge("normalize", "respond")
    workflow.add_edge("respond", END)
    return workflow.compile()


graph = build_graph()


if __name__ == "__main__":
    input_text = input("Enter your text: ")

    result = graph.invoke({"input_text": input_text})
    
    print("-" * 100)
    print(result.get("output"))
    print("-" * 100)
