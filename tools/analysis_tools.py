import math
import datetime
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports standard math operations and functions like sqrt, log, pow. Example: '1500 * 0.15' or 'sqrt(144)'."""
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"🧮 Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_date_context(_: str = "") -> str:
    """Get the current date, time, day of the week, and quarter. Useful for providing time-aware financial or research context."""
    now = datetime.datetime.now()
    quarter = (now.month - 1) // 3 + 1
    return (
        f"📅 Current Date: {now.strftime('%B %d, %Y')}\n"
        f"  Time: {now.strftime('%H:%M:%S')}\n"
        f"  Day: {now.strftime('%A')}\n"
        f"  Quarter: Q{quarter} {now.year}"
    )

@tool
def summarize_text(text: str) -> str:
    """Summarize a long piece of text into a concise paragraph. Useful after gathering research or financial data."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke([HumanMessage(content=f"Summarize the following concisely in 3-5 sentences:\n\n{text}")])
        return f"📝 Summary:\n{response.content}"
    except Exception as e:
        return f"Summarization error: {str(e)}"