from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from agents.state import AgentState, keep_last
from tools import RESEARCH_TOOLS

SYSTEM_PROMPT = (
    "You are a research specialist. Find news, background context, and factual information. "
    "Use web_search for recent news articles and current events about companies or topics. "
    "Use wikipedia_search for company history, founding, products, and established facts. "
    "Scope: news headlines, company background, industry context, product info, recent events. "
    "Do NOT look up live stock prices, market caps, P/E ratios, or exchange rates — those come from dedicated financial tools. "
    "Present your findings concisely without asking follow-up questions or commenting on what you cannot do."
)

def research_agent_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(RESEARCH_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + keep_last(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "active_agents": state.get("active_agents", []) + ["research_agent"],
    }
