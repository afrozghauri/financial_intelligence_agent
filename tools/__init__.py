from .search_tools import get_tavily_tool, wikipedia_search
from .finance_tools import get_stock_price, get_crypto_price, get_exchange_rate
from .analysis_tools import calculator, get_date_context, summarize_text
from .report_tools import save_report_to_file, format_markdown_report
from .rag_tools import index_document, search_knowledge_base

RESEARCH_TOOLS = [get_tavily_tool(), wikipedia_search]
FINANCE_TOOLS = [get_stock_price, get_crypto_price, get_exchange_rate]
ANALYSIS_TOOLS = [calculator, get_date_context, summarize_text]
REPORT_TOOLS = [format_markdown_report, save_report_to_file]
RAG_TOOLS = [index_document, search_knowledge_base]
ALL_TOOLS = RESEARCH_TOOLS + FINANCE_TOOLS + ANALYSIS_TOOLS + REPORT_TOOLS + RAG_TOOLS
