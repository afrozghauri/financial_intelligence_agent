from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from agents.state import AgentState, keep_last
from tools import RAG_TOOLS

SYSTEM_PROMPT = (
    "You are a knowledge management specialist with access to a persistent semantic vector database. "
    "Use search_knowledge_base to retrieve relevant findings from past reports and analyses — "
    "always search before answering questions about prior work to give accurate, consistent information. "
    "Use index_document to store important summaries or findings so they can be recalled later. "
    "Clearly state what you found or indexed, and explain its relevance to the current question."
)

def rag_agent_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(RAG_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + keep_last(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "active_agents": state.get("active_agents", []) + ["rag_agent"],
    }
