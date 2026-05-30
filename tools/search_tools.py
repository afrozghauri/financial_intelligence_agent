import wikipedia
from langchain.tools import tool
from langchain_tavily import TavilySearch

def get_tavily_tool():
    return TavilySearch(
        max_results=4,
        description="Search the internet for recent news, events, or factual information about companies, people, or topics."
    )

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for background information on a topic, company, or person."""
    try:
        result = wikipedia.summary(query, sentences=5, auto_suggest=True)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation: {e.options[:5]}"
    except Exception as e:
        return f"Wikipedia error: {str(e)}"