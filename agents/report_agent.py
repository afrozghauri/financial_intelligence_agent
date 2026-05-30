from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from agents.state import AgentState, keep_last
from tools import REPORT_TOOLS

SYSTEM_PROMPT = (
    "You are a reporting specialist who creates professional financial intelligence reports. "
    "Use format_markdown_report to structure raw data into a clean, professional Markdown document "
    "with sections: Executive Summary, Key Financial Data, Market Analysis, News Highlights, and Conclusion. "
    "Use save_report_to_file to persist the formatted report to the reports/ directory. "
    "Always format first, then save. Ensure reports are comprehensive, accurate, and professionally written."
)

def report_agent_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(REPORT_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + keep_last(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "active_agents": state.get("active_agents", []) + ["report_agent"],
    }
