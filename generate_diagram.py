"""Run this script to regenerate diagram.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

FIG_BG = "#f0f4f8"
COLORS = {
    "user":       "#1a237e",
    "supervisor": "#c62828",
    "research":   "#2e7d32",
    "finance":    "#1565c0",
    "analysis":   "#6a1b9a",
    "report":     "#e65100",
    "rag":        "#00695c",
    "hitl":       "#f57f17",
    "output":     "#004d40",
}

def box(ax, cx, cy, w, h, text, color, fontsize=9, alpha=1.0, text_color="white"):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.07",
        facecolor=color, edgecolor="white", linewidth=1.6,
        alpha=alpha, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4, multialignment="center")

def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.4, ls="solid"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=ls),
        zorder=2,
    )

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(19, 11))
ax.set_xlim(0, 19)
ax.set_ylim(0, 11)
ax.axis("off")
ax.set_facecolor(FIG_BG)
fig.patch.set_facecolor(FIG_BG)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(9.5, 10.6, "Financial Intelligence Agent — Multi-Agent Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1a237e")

# ── LangGraph boundary ────────────────────────────────────────────────────────
lg = FancyBboxPatch((0.3, 1.1), 18.4, 8.3,
                    boxstyle="round,pad=0.15",
                    facecolor="#e8eaf6", edgecolor="#7986cb", linewidth=2,
                    linestyle="--", zorder=1)
ax.add_patch(lg)
ax.text(9.5, 9.1, "LangGraph StateGraph  ·  SQLite Persistent Checkpointing  ·  interrupt_before=[\"report_save_tools\"]",
        ha="center", va="center", fontsize=8.5, color="#5c6bc0", style="italic")

# ── User / Gradio ─────────────────────────────────────────────────────────────
box(ax, 9.5, 10.1, 5.0, 0.65, "User  ·  Gradio Web UI", COLORS["user"], fontsize=10)

# ── Supervisor ────────────────────────────────────────────────────────────────
box(ax, 9.5, 7.7, 4.2, 0.75,
    "Supervisor\nGPT-4o-mini · Routes tasks to agents",
    COLORS["supervisor"], fontsize=9)
arrow(ax, 9.5, 9.77, 9.5, 8.08)

# ── Agents (5) ────────────────────────────────────────────────────────────────
# Spread across: research=1.8, finance=5.1, analysis=8.5, report=12.2, rag=16.0
AGENT_Y = 6.0
agents = [
    (1.8,  AGENT_Y, "Research\nAgent",  COLORS["research"]),
    (5.1,  AGENT_Y, "Finance\nAgent",   COLORS["finance"]),
    (8.5,  AGENT_Y, "Analysis\nAgent",  COLORS["analysis"]),
    (12.2, AGENT_Y, "Report\nAgent",    COLORS["report"]),
    (16.0, AGENT_Y, "RAG\nAgent",       COLORS["rag"]),
]
AW, AH = 2.7, 0.85
for cx, cy, label, color in agents:
    box(ax, cx, cy, AW, AH, label, color, fontsize=9)
    arrow(ax, 9.5, 7.33, cx, cy + AH / 2)           # supervisor → agent
    arrow(ax, cx, cy + AH / 2, 9.5, 7.33,           # agent → supervisor (return)
          color="#aaaaaa", lw=1.0, ls="dashed")

# ── Standard tool boxes: research, finance, analysis, rag ────────────────────
TY = 4.1   # centre y for tool boxes
TW, TH = 2.7, 1.0

standard = [
    (1.8,  TY,      "web_search\nwikipedia_search",              COLORS["research"]),
    (5.1,  TY - 0.1,"get_stock_price\nget_crypto_price\nget_exchange_rate", COLORS["finance"]),
    (8.5,  TY,      "calculator\nget_date_context\nsummarize_text",        COLORS["analysis"]),
    (16.0, TY,      "search_knowledge_base\nindex_document",     COLORS["rag"]),
]
for (cx, cy, label, color), (acx, acy, _, _) in zip(
        standard, [a for a in agents if a[0] != 12.2]):
    box(ax, cx, cy, TW, TH, label, color, fontsize=7.5, alpha=0.80)
    arrow(ax, acx, acy - AH / 2, cx, cy + TH / 2)          # agent → tools
    arrow(ax, cx + 0.1, cy + TH / 2, acx + 0.1, acy - AH / 2,
          color="#bbbbbb", lw=1.0, ls="dashed")             # tools → agent

# ── Report agent: 3-tier (format → HITL → save) ──────────────────────────────
RX = 12.2
RF_Y = 5.0    # format tool
HI_Y = 4.1   # HITL approval
RS_Y = 3.2   # save tool
RTH  = 0.6   # tier height
RTW  = 2.7

# Arrows: report_agent → format → HITL → save → report_agent
arrow(ax, RX, AGENT_Y - AH / 2, RX, RF_Y + RTH / 2)              # agent → format
box(ax, RX, RF_Y, RTW, RTH, "format_markdown_report", COLORS["report"], fontsize=7.5, alpha=0.82)
arrow(ax, RX, RF_Y - RTH / 2, RX, HI_Y + RTH / 2)                # format → HITL
box(ax, RX, HI_Y, RTW, RTH,
    "[ HUMAN APPROVAL ]\ninterrupt_before save",
    COLORS["hitl"], fontsize=7.5)
arrow(ax, RX, HI_Y - RTH / 2, RX, RS_Y + RTH / 2)                # HITL → save
box(ax, RX, RS_Y, RTW, RTH, "save_report_to_file", COLORS["report"], fontsize=7.5, alpha=0.82)
arrow(ax, RX + 0.1, RS_Y + RTH / 2, RX + 0.1, AGENT_Y - AH / 2, # save → agent
      color="#bbbbbb", lw=1.0, ls="dashed")

# ── ChromaDB annotation for RAG ───────────────────────────────────────────────
box(ax, 16.0, 2.85, 2.7, 0.55,
    "ChromaDB  ·  OpenAI Embeddings", "#004d40", fontsize=7.0, alpha=0.75)
arrow(ax, 16.0, 3.6, 16.0, 3.13)

# ── External APIs label ───────────────────────────────────────────────────────
ax.text(9.5, 2.3,
        "External APIs:  yfinance  ·  CoinGecko  ·  ExchangeRate-API  ·  "
        "Tavily Search  ·  Wikipedia  ·  OpenAI GPT-4o-mini  ·  ChromaDB (local)",
        ha="center", va="center", fontsize=8, color="#546e7a", style="italic")

# ── Final output ──────────────────────────────────────────────────────────────
box(ax, 9.5, 1.55, 5.0, 0.65, "Final Response  →  Gradio Chat", COLORS["output"], fontsize=9)

# FINISH path: supervisor → output (dashed)
arrow(ax, 8.15, 7.33, 9.0, 1.88, color=COLORS["supervisor"], lw=1.2, ls="dashed")
ax.text(7.9, 4.6, "FINISH", fontsize=7.5, color=COLORS["supervisor"],
        rotation=80, style="italic")

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    (COLORS["supervisor"], "Supervisor"),
    (COLORS["research"],   "Research Agent"),
    (COLORS["finance"],    "Finance Agent"),
    (COLORS["analysis"],   "Analysis Agent"),
    (COLORS["report"],     "Report Agent"),
    (COLORS["rag"],        "RAG Agent"),
    (COLORS["hitl"],       "Human Approval (HITL)"),
]
handles = [mpatches.Patch(facecolor=c, label=l, edgecolor="white") for c, l in legend_items]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.0),
          fontsize=8, framealpha=0.88, edgecolor="#90a4ae", ncol=1)

plt.tight_layout(pad=0.3)
plt.savefig("diagram.png", dpi=150, bbox_inches="tight", facecolor=FIG_BG)
print("diagram.png saved successfully.")
