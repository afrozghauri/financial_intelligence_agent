from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.tools import tool

CHROMA_DIR = "./chroma_db"
COLLECTION = "financial_knowledge"

def _get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_DIR,
    )

@tool
def index_document(content: str, source: str = "agent") -> str:
    """Index a piece of text into the persistent knowledge base for future retrieval.
    Use this to store research findings, financial summaries, or report content.
    Provide the full text as `content` and a short label as `source` (e.g. 'tesla_report')."""
    try:
        vs = _get_vectorstore()
        vs.add_documents([Document(page_content=content, metadata={"source": source})])
        return f"✅ Indexed into knowledge base (source: '{source}', {len(content):,} chars)"
    except Exception as e:
        return f"Indexing error: {str(e)}"

@tool
def search_knowledge_base(query: str) -> str:
    """Search the persistent knowledge base using semantic similarity.
    Returns the most relevant past research, analyses, and report summaries.
    Use this to recall earlier findings or check whether a topic has been analysed before."""
    try:
        vs = _get_vectorstore()
        results = vs.similarity_search_with_relevance_scores(query, k=3)
        if not results:
            return "📚 Knowledge base is empty — no documents have been indexed yet."
        output = "📚 Knowledge Base Results:\n\n"
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:400].strip().replace("\n", " ")
            output += f"**[{i}] source=`{source}` | relevance={score:.2f}**\n{snippet}…\n\n"
        return output.strip()
    except Exception as e:
        return f"Knowledge base search error: {str(e)}"
