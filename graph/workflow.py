from dotenv import load_dotenv
load_dotenv()  # must run before project imports that read env vars at module load time

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage

from agents.state import AgentState
from agents.supervisor import supervisor_node
from agents.research_agent import research_agent_node
from agents.finance_agent import finance_agent_node
from agents.analysis_agent import analysis_agent_node
from agents.report_agent import report_agent_node
from agents.rag_agent import rag_agent_node
from tools import RESEARCH_TOOLS, FINANCE_TOOLS, ANALYSIS_TOOLS, RAG_TOOLS
from tools.report_tools import format_markdown_report, save_report_to_file

load_dotenv()

# Split report tools so the interrupt can be placed between format and save.
REPORT_FORMAT_TOOLS = [format_markdown_report]
REPORT_SAVE_TOOLS = [save_report_to_file]

def _route_report_agent(state: AgentState) -> str:
    """Direct report_agent to the correct tool node based on which tool it called."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return END
    tool_name = last.tool_calls[0]["name"]
    if tool_name == "format_markdown_report":
        return "report_format_tools"
    if tool_name == "save_report_to_file":
        return "report_save_tools"   # graph pauses here (interrupt_before)
    return END

def build_graph(memory):
    graph = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)

    graph.add_node("research_agent", research_agent_node)
    graph.add_node("research_tools", ToolNode(RESEARCH_TOOLS))

    graph.add_node("finance_agent", finance_agent_node)
    graph.add_node("finance_tools", ToolNode(FINANCE_TOOLS))

    graph.add_node("analysis_agent", analysis_agent_node)
    graph.add_node("analysis_tools", ToolNode(ANALYSIS_TOOLS))

    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("rag_tools", ToolNode(RAG_TOOLS))

    # Report agent has two separate tool nodes so human approval sits between them.
    graph.add_node("report_agent", report_agent_node)
    graph.add_node("report_format_tools", ToolNode(REPORT_FORMAT_TOOLS))
    graph.add_node("report_save_tools",   ToolNode(REPORT_SAVE_TOOLS))

    # ── Entry ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── Supervisor routing ────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", "FINISH"),
        {
            "research_agent": "research_agent",
            "finance_agent":  "finance_agent",
            "analysis_agent": "analysis_agent",
            "report_agent":   "report_agent",
            "rag_agent":      "rag_agent",
            "FINISH":         END,
        },
    )

    # ── Standard agents: call tools or return to supervisor ───────────────────
    for agent, tool_node in [
        ("research_agent", "research_tools"),
        ("finance_agent",  "finance_tools"),
        ("analysis_agent", "analysis_tools"),
        ("rag_agent",      "rag_tools"),
    ]:
        graph.add_conditional_edges(agent, tools_condition, {
            "tools": tool_node,
            END: "supervisor",
        })
        graph.add_edge(tool_node, agent)

    # ── Report agent: custom routing splits format vs. save tool nodes ─────────
    graph.add_conditional_edges("report_agent", _route_report_agent, {
        "report_format_tools": "report_format_tools",
        "report_save_tools":   "report_save_tools",   # interrupted here
        END: "supervisor",
    })
    graph.add_edge("report_format_tools", "report_agent")
    graph.add_edge("report_save_tools",   "report_agent")

    # ── Compile with HITL interrupt before the save step ─────────────────────
    return graph.compile(
        checkpointer=memory,
        interrupt_before=["report_save_tools"],
    )
