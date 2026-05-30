from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from agents.state import AgentState, keep_last
from tools import ANALYSIS_TOOLS

SYSTEM_PROMPT = (
    "You are an analytical specialist who processes and interprets financial and research data. "
    "Use calculator for arithmetic, percentage calculations, and mathematical operations — "
    "it supports standard math functions like sqrt, log, and pow. "
    "Use get_date_context to provide current date, time, day of week, and fiscal quarter for time-sensitive analysis. "
    "Use summarize_text to condense lengthy research findings into concise, actionable summaries. "
    "Explain the significance of your calculations and provide clear interpretations."
)

def analysis_agent_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(ANALYSIS_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + keep_last(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "active_agents": state.get("active_agents", []) + ["analysis_agent"],
    }
