from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from agents.state import AgentState, keep_last
from tools import FINANCE_TOOLS

SYSTEM_PROMPT = (
    "You are a financial data retrieval specialist. Fetch the requested live market data and return it as-is. "
    "Use get_stock_price for equity data: price, market cap, P/E ratio, 52-week range. "
    "Use get_crypto_price for cryptocurrency price and 24h market data. "
    "Use get_exchange_rate for currency pair rates. "
    "Return the raw fetched numbers clearly. Do NOT perform calculations, do NOT convert currencies yourself, "
    "and do NOT write summaries or analysis — just fetch and return the data."
)

def finance_agent_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(FINANCE_TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + keep_last(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "active_agents": state.get("active_agents", []) + ["finance_agent"],
    }
