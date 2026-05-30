from dotenv import load_dotenv
load_dotenv()  # load .env before any project imports that read env vars at module load time

import gradio as gr
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from graph.workflow import build_graph
from graph.memory import get_memory

memory = get_memory()
graph = build_graph(memory)

EXAMPLE_PROMPTS = [
    "Research Tesla, get its current stock price, convert the market cap to EUR, and save a full report.",
    "What is Bitcoin's current price and market cap? Summarize recent Bitcoin news and save a report.",
    "Compare Apple and Microsoft stock prices. Which has a higher P/E ratio? Do the math and summarize.",
    "Search for the latest AI regulation news in Europe, get today's date context, and format a briefing report.",
    "What is 15% of Tesla's current stock price? Also get the USD to JPY exchange rate.",
    "Search the knowledge base for any past analyses on Apple or AAPL.",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}

def _last_text(messages: list) -> str:
    return next(
        (m.content for m in reversed(messages)
         if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None)),
        "Task complete.",
    )

def format_agent_trace(messages: list) -> str:
    trace = []
    for m in messages:
        role = type(m).__name__
        if role == "HumanMessage":
            continue
        elif role == "AIMessage":
            if m.tool_calls:
                for tc in m.tool_calls:
                    trace.append(f"🔧 **Tool Call:** `{tc['name']}` — args: `{tc['args']}`")
            elif m.content:
                trace.append(f"🤖 **Agent Response:**\n{m.content}")
        elif role == "ToolMessage":
            trace.append(f"📦 **Tool Result ({m.name}):**\n```\n{m.content[:500]}\n```")
    return "\n\n".join(trace)

# ── Core chat handler ─────────────────────────────────────────────────────────

def chat(user_message, history, thread_id, show_trace):
    if not user_message.strip():
        return history, history, thread_id, gr.update(visible=False)

    if not thread_id:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=user_message)], "active_agents": []}

    try:
        result = graph.invoke(inputs, config=config)
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            msg = "⚠️ OpenAI rate limit reached. Please wait a few seconds and try again."
        else:
            msg = f"⚠️ An error occurred: {err[:300]}"
        history = history + [_msg("user", user_message), _msg("assistant", msg)]
        return history, history, thread_id, gr.update(visible=False)

    messages = result["messages"]

    # ── Check for HITL interrupt (graph paused before report_save_tools) ──────
    state = graph.get_state(config)
    if state.next and "report_save_tools" in state.next:
        formatted = next(
            (m.content for m in reversed(messages)
             if isinstance(m, ToolMessage) and getattr(m, "name", "") == "format_markdown_report"),
            _last_text(messages),
        )
        display = (
            "📋 **Report formatted — awaiting your approval before saving.**\n\n"
            f"{formatted}\n\n"
            "---\n*Use the **Human Approval** panel on the right to approve or cancel.*"
        )
        history = history + [_msg("user", user_message), _msg("assistant", display)]
        return history, history, thread_id, gr.update(visible=True)

    # ── Normal completion ─────────────────────────────────────────────────────
    agents_used = result.get("active_agents", [])
    summary = (
        f"\n\n---\n🧠 **Agents invoked:** "
        f"{', '.join(dict.fromkeys(agents_used)) if agents_used else 'supervisor only'}"
    )
    display = _last_text(messages) + summary

    if show_trace:
        display += f"\n\n---\n### 🔍 Agent Reasoning Trace\n{format_agent_trace(messages)}"

    history = history + [_msg("user", user_message), _msg("assistant", display)]
    return history, history, thread_id, gr.update(visible=False)

# ── HITL: approve → resume graph, execute the save tool ──────────────────────

def approve_report(history, thread_id, show_trace):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(None, config=config)
    messages = result["messages"]

    agents_used = result.get("active_agents", [])
    summary = (
        f"\n\n---\n🧠 **Agents invoked:** "
        f"{', '.join(dict.fromkeys(agents_used)) if agents_used else 'supervisor only'}"
    )
    display = "✅ **Report approved and saved.**\n\n" + _last_text(messages) + summary

    if show_trace:
        display += f"\n\n---\n### 🔍 Agent Reasoning Trace\n{format_agent_trace(messages)}"

    history = history + [_msg("user", "✅ Approve & save report"), _msg("assistant", display)]
    return history, history, gr.update(visible=False)

# ── HITL: reject → inject cancellation message, advance graph past interrupt ──

def reject_report(history, thread_id):
    config = {"configurable": {"thread_id": thread_id}}

    state = graph.get_state(config)
    last = state.values["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        rejection = ToolMessage(
            content="Report saving was cancelled by the user. Do not attempt to save again.",
            tool_call_id=last.tool_calls[0]["id"],
            name="save_report_to_file",
        )
        graph.update_state(config, {"messages": [rejection]}, as_node="report_save_tools")

    graph.invoke(None, config=config)

    history = history + [
        _msg("user", "❌ Cancel saving report"),
        _msg("assistant", "Report saving cancelled. The formatted report was not written to disk."),
    ]
    return history, history, gr.update(visible=False)

# ── Session management ────────────────────────────────────────────────────────

def new_session():
    return [], [], str(uuid.uuid4())

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Financial Intelligence Agent") as demo:
    gr.Markdown("""
    # 📊 Financial & Market Intelligence Platform
    ### Powered by LangChain · LangGraph · Multi-Agent Supervisor Architecture
    **Agents:** Research · Finance · Analysis · Report · RAG
    | **Memory:** SQLite Checkpointing · ChromaDB Vector Store
    """)

    thread_id_state    = gr.State(str(uuid.uuid4()))
    chat_history_state = gr.State([])

    with gr.Row():
        # ── Left: chat ────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=520, label="Multi-Agent Conversation"
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="e.g. Research NVIDIA, get its stock price, and save a report...",
                    label="Your Query",
                    scale=5,
                )
                send_btn = gr.Button("▶ Run", variant="primary", scale=1)

        # ── Right: controls ───────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Controls")
            show_trace   = gr.Checkbox(label="Show Agent Reasoning Trace", value=False)
            new_chat_btn = gr.Button("🔄 New Session", variant="secondary")
            session_label = gr.Textbox(label="Session Thread ID", interactive=False)

            with gr.Group(visible=False) as approval_panel:
                gr.Markdown("### 🔐 Human Approval Required")
                gr.Markdown(
                    "The report has been formatted. Approve to write it to disk, "
                    "or cancel to discard."
                )
                approve_btn = gr.Button("✅ Approve & Save", variant="primary")
                reject_btn  = gr.Button("❌ Cancel Saving",  variant="stop")

            gr.Markdown("### 💡 Example Queries")
            for prompt in EXAMPLE_PROMPTS:
                gr.Button(prompt[:60] + "…").click(
                    fn=lambda p=prompt: p, outputs=user_input
                )

    # ── Wiring ────────────────────────────────────────────────────────────────
    chat_outputs = [chatbot, chat_history_state, thread_id_state, approval_panel]

    send_btn.click(
        fn=chat,
        inputs=[user_input, chat_history_state, thread_id_state, show_trace],
        outputs=chat_outputs,
    ).then(lambda: "", outputs=user_input)

    user_input.submit(
        fn=chat,
        inputs=[user_input, chat_history_state, thread_id_state, show_trace],
        outputs=chat_outputs,
    ).then(lambda: "", outputs=user_input)

    approve_btn.click(
        fn=approve_report,
        inputs=[chat_history_state, thread_id_state, show_trace],
        outputs=[chatbot, chat_history_state, approval_panel],
    )

    reject_btn.click(
        fn=reject_report,
        inputs=[chat_history_state, thread_id_state],
        outputs=[chatbot, chat_history_state, approval_panel],
    )

    new_chat_btn.click(
        fn=new_session,
        outputs=[chatbot, chat_history_state, thread_id_state],
    )

    thread_id_state.change(fn=lambda t: t, inputs=thread_id_state, outputs=session_label)
    demo.load(fn=lambda t: t, inputs=thread_id_state, outputs=session_label)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Glass())
