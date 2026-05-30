from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str                  # which agent the supervisor routes to next
    final_report: str          # accumulated final report content
    active_agents: list[str]   # tracks which agents have been called

def keep_last(messages: list, n: int = 20) -> list:
    """Return recent messages starting at a clean HumanMessage boundary.

    Slicing at an arbitrary offset can leave orphaned ToolMessages (no preceding
    tool_calls AIMessage), which causes an OpenAI 400 error.  We trim to the last
    n messages then advance to the first HumanMessage so the slice is always valid.
    """
    if len(messages) <= n:
        return messages
    trimmed = messages[-n:]
    # Preferred: start at a HumanMessage (clean turn boundary)
    for i, msg in enumerate(trimmed):
        if type(msg).__name__ == "HumanMessage":
            return trimmed[i:]
    # Fallback: skip any leading ToolMessages to avoid the orphan error
    for i, msg in enumerate(trimmed):
        if type(msg).__name__ != "ToolMessage":
            return trimmed[i:]
    return trimmed