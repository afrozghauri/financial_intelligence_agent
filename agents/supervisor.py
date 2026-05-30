from typing import Literal
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from agents.state import AgentState, keep_last

AgentName = Literal[
    "research_agent",
    "finance_agent",
    "analysis_agent",
    "report_agent",
    "rag_agent",
    "FINISH",
]

class Route(BaseModel):
    next: AgentName

SUPERVISOR_PROMPT = """You are a task router for a Financial Intelligence Platform.
Your job: pick the NEXT agent to run, or FINISH when everything is done.

AGENTS:
- research_agent  → search the INTERNET for news, recent events, company background (use for "search for news", "find articles", "latest news")
- finance_agent   → live stock price, market cap, P/E ratio, crypto price, exchange rate
- analysis_agent  → math, calculations, percentage, conversions, date context, summarization
- report_agent    → format all gathered data into an HTML report and save it
- rag_agent       → search or index the INTERNAL knowledge base of PAST SAVED REPORTS (only when user says "search past analyses", "recall previous report", "what did we analyse before")

CRITICAL DISTINCTION:
- "search for news / find articles / latest news / look up information" → research_agent (internet search)
- "search past analyses / recall previous report / knowledge base" → rag_agent (internal DB)
- NEVER send internet news searches to rag_agent

ROUTING EXAMPLES (follow these patterns exactly):

"Get Apple stock price"
→ finance_agent → FINISH

"Research Tesla news and background"
→ research_agent → FINISH

"Search for latest AI regulation news in Europe, get date context, save a briefing report"
→ research_agent → analysis_agent → report_agent → FINISH

"Get Bitcoin price and save a report"
→ finance_agent → report_agent → FINISH

"Get Tesla stock price, convert market cap to EUR, save a report"
→ finance_agent → analysis_agent → report_agent → FINISH

"Research Apple, get stock price, save a report"
→ research_agent → finance_agent → report_agent → FINISH

"Compare Apple vs Microsoft stocks, save a report"
→ finance_agent → report_agent → FINISH

"Research Tesla, get stock price, convert market cap to EUR, save a full report"
→ research_agent → finance_agent → analysis_agent → report_agent → FINISH

"Search past analyses on Apple" or "what have we analysed before"
→ rag_agent → FINISH

RULES:
- Always check what tasks the user asked for and which are still pending in the conversation.
- If the user asked to "save", "report", "briefing", or "document" → route to report_agent as the LAST step.
- NEVER output FINISH while report_agent is still pending.
- If a previous agent could not complete a task, route to the correct agent that can."""

def supervisor_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured = llm.with_structured_output(Route)
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + keep_last(state["messages"], n=20)
    try:
        response = structured.invoke(messages)
        return {"next": response.next}
    except Exception:
        return {"next": "FINISH"}
