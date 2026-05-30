---
title: Financial Intelligence Agent
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Financial Intelligence Agent

A multi-agent AI platform for financial research, live market data, quantitative analysis, and automated report generation — built on **LangGraph**'s supervisor pattern with a **Gradio** web UI.

![Architecture Diagram](diagram.png)

---

## Architecture

The system uses a **Supervisor → Specialized Agent** pattern. A central supervisor LLM reads every user message and decides which specialist agent to invoke. Agents call their tools, return results to the supervisor, and the loop continues until the task is complete.

```
User (Gradio UI)
      │
      ▼
 Supervisor  (GPT-4o-mini · routes tasks)
  ┌───┼───────┬──────────┬────────┐
  ▼   ▼       ▼          ▼        ▼
Research  Finance  Analysis  Report  RAG
 Agent    Agent    Agent     Agent   Agent
  │         │        │         │       │
Tools     Tools    Tools   Format  index_doc
                           ──↓──   search_kb
                         [HITL]
                           ──↓──
                          Save
```

### Agents

| Agent | Role | Tools |
|---|---|---|
| **Supervisor** | Routes requests to the best-fit agent; signals `FINISH` when done | — |
| **Research Agent** | Finds news, background info, and recent events | `web_search`, `wikipedia_search` |
| **Finance Agent** | Retrieves live market data | `get_stock_price`, `get_crypto_price`, `get_exchange_rate` |
| **Analysis Agent** | Performs calculations, date context, text summarization | `calculator`, `get_date_context`, `summarize_text` |
| **Report Agent** | Formats reports then waits for human approval before saving | `format_markdown_report`, `save_report_to_file` |
| **RAG Agent** | Searches and indexes the persistent vector knowledge base | `search_knowledge_base`, `index_document` |

### Tools

| Tool | Description | Data Source |
|---|---|---|
| `web_search` | Real-time internet search (4 results) | Tavily API |
| `wikipedia_search` | Wikipedia summaries with disambiguation handling | Wikipedia |
| `get_stock_price` | Price, market cap, P/E ratio, 52-week range | yfinance |
| `get_crypto_price` | Price, market cap, 24h change | CoinGecko API |
| `get_exchange_rate` | Currency pair conversion rates | ExchangeRate-API |
| `calculator` | Safe math evaluation (`sqrt`, `log`, `pow`, etc.) | Python `math` |
| `get_date_context` | Current date, time, day, and fiscal quarter | System clock |
| `summarize_text` | Condenses text into a 3-5 sentence summary | GPT-4o-mini |
| `format_markdown_report` | Structures raw data into a professional Markdown report | GPT-4o-mini |
| `save_report_to_file` | Saves report to `reports/` — requires human approval first | Local filesystem |
| `search_knowledge_base` | Semantic search over past analyses and reports | ChromaDB + OpenAI Embeddings |
| `index_document` | Indexes text into the vector store for future retrieval | ChromaDB + OpenAI Embeddings |

---

## Features

### RAG — Persistent Vector Knowledge Base
The **RAG Agent** stores and retrieves information from a local ChromaDB vector database (`chroma_db/`). Every saved report or key finding can be indexed, enabling the system to:
- Recall past analyses across sessions
- Avoid duplicate research
- Find related context from previous work

### Human-in-the-Loop Report Approval
Before any report is written to disk, the graph **pauses** (via LangGraph's `interrupt_before`) and surfaces an approval panel in the UI. You can:
- **Approve & Save** — resumes the graph, writes the file to `reports/`
- **Cancel Saving** — injects a rejection message, the graph completes without saving

---

## Setup

### Prerequisites

- Python 3.9+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Tavily API key](https://tavily.com/)
- [ExchangeRate-API key](https://www.exchangerate-api.com/)
- *(Optional)* [LangSmith API key](https://smith.langchain.com/) for tracing

### Installation

```bash
git clone <repo-url>
cd financial_intelligence_agent
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
EXCHANGE_RATE_API_KEY=...

# Optional: enables LangSmith tracing
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=financial-intelligence-agent
```

For **Hugging Face Spaces**, set these as repository secrets (Settings → Variables and secrets) instead of using a `.env` file.

---

## Usage

```bash
python app.py
```

Open the Gradio UI at `http://localhost:7860`.

### Example Queries

| Query | Agents Typically Used |
|---|---|
| Research Tesla, get stock price, convert market cap to EUR, save a report | Research + Finance + Analysis + Report |
| What is Bitcoin's price? Summarize recent news and save a report | Finance + Research + Analysis + Report |
| Compare Apple vs Microsoft stocks — which has a higher P/E? | Finance + Analysis |
| Search for the latest AI regulation news and format a briefing | Research + Report |
| What is 15% of Tesla's current stock price? | Finance + Analysis |
| Search the knowledge base for past analyses on Apple | RAG |

### UI Controls

| Control | Description |
|---|---|
| **Show Agent Reasoning Trace** | Expands all tool calls and raw results below the answer |
| **New Session** | Starts a fresh conversation with a new thread ID |
| **Session Thread ID** | Keep this to resume a conversation with full memory |
| **Approve & Save** | (appears when a report is ready) Confirms writing to disk |
| **Cancel Saving** | (appears when a report is ready) Discards the save without losing the formatted content |

---

## Memory and Persistence

| Store | Technology | Location | Persists across restarts |
|---|---|---|---|
| Conversation memory | SQLite via LangGraph `SqliteSaver` | `agent_memory.db` | Yes |
| Knowledge base (RAG) | ChromaDB + OpenAI Embeddings | `chroma_db/` | Yes |
| Generated reports | Markdown files | `reports/` | Yes |

> **Note for Hugging Face Spaces:** the Spaces filesystem is ephemeral and resets on container restart. For persistent storage on Spaces, mount a [Persistent Storage](https://huggingface.co/docs/hub/spaces-storage) volume or use an external database.

---

## Project Structure

```
financial_intelligence_agent/
├── app.py                    # Gradio web interface — entry point
├── requirements.txt
├── .env                      # API keys (not committed — see .gitignore)
├── .gitignore
├── diagram.png               # Architecture diagram
├── generate_diagram.py       # Script to regenerate diagram.png
│
├── agents/
│   ├── __init__.py           # Exports all agent nodes
│   ├── state.py              # Shared AgentState TypedDict
│   ├── supervisor.py         # Supervisor routing logic
│   ├── research_agent.py     # Web search + Wikipedia
│   ├── finance_agent.py      # Stock, crypto, FX data
│   ├── analysis_agent.py     # Calculator, date, summarize
│   ├── report_agent.py       # Format + save reports (HITL gated)
│   └── rag_agent.py          # Vector store search + index
│
├── graph/
│   ├── __init__.py           # Exports build_graph, get_memory
│   ├── workflow.py           # LangGraph StateGraph assembly + interrupt
│   └── memory.py             # SQLite checkpointer
│
├── tools/
│   ├── __init__.py           # Aggregates all tools by category
│   ├── search_tools.py       # web_search, wikipedia_search
│   ├── finance_tools.py      # get_stock_price, get_crypto_price, get_exchange_rate
│   ├── analysis_tools.py     # calculator, get_date_context, summarize_text
│   ├── report_tools.py       # format_markdown_report, save_report_to_file
│   └── rag_tools.py          # index_document, search_knowledge_base
│
├── reports/                  # Auto-created; timestamped .md reports
├── chroma_db/                # Auto-created; ChromaDB vector store
└── agent_memory.db           # Auto-created; SQLite conversation memory
```

---

## Deploying to Hugging Face Spaces

This repo is pre-configured for Spaces deployment (see YAML frontmatter above and `server_name="0.0.0.0"` in `app.py`).

**Steps (requires you to do once):**
1. Push this repo to a GitHub repository
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space) → choose **Gradio** SDK → link your GitHub repo
3. In Space Settings → **Variables and secrets**, add your API keys:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
   - `EXCHANGE_RATE_API_KEY`
4. *(Optional)* Add a Persistent Storage volume so `chroma_db/`, `reports/`, and `agent_memory.db` survive container restarts

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-4o-mini |
| Agent framework | LangChain + LangGraph |
| Vector store (RAG) | ChromaDB + OpenAI Embeddings (`text-embedding-3-small`) |
| Web UI | Gradio |
| Conversation memory | LangGraph SQLite checkpointer |
| Web search | Tavily |
| Stock data | yfinance |
| Crypto data | CoinGecko API |
| Currency data | ExchangeRate-API |
| Tracing | LangSmith (optional) |
