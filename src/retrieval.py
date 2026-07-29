"""
retrieval.py
Queries the Pinecone vector store built by database.py and returns the
most relevant chunks of IPC / BNS text for a given question or section number.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_INDEX_NAME = "nyaya-flow"

# Module-level cache so we don't reconnect / re-embed on every call.
_vectorstore = None


def get_vectorstore(index_name: str = _INDEX_NAME) -> PineconeVectorStore:
    """Lazily connects to the existing Pinecone index (assumes database.py has already run)."""
    global _vectorstore
    if _vectorstore is None:
        if not os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY") == "your_actual_api_key_here":
            raise EnvironmentError(
                "PINECONE_API_KEY is missing or still a placeholder. "
                "Set a real key in your .env file before running retrieval."
            )
        embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)
        _vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return _vectorstore


def retrieve_chunks(query: str, k: int = 4, law_source: str | None = None):
    """
    Returns the top-k most relevant chunks for a query.

    Args:
        query: natural-language question or a section reference (e.g. "Section 302 IPC").
        k: number of chunks to return.
        law_source: optional filter, e.g. "Indian Penal Code 1860 official PDF"
                    or "Bhartiya Nyaya Sanhita 2023 PDF", to restrict the search
                    to one statute only.
    """
    vectorstore = get_vectorstore()
    search_kwargs = {"k": k}
    if law_source:
        search_kwargs["filter"] = {"law_source": law_source}

    results = vectorstore.similarity_search(query, **search_kwargs)
    return results


def format_chunks_for_prompt(chunks) -> str:
    """Turns retrieved Document objects into a clean context block for the LLM."""
    if not chunks:
        return "No relevant sections found."

    formatted = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("law_source", "Unknown source")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[{i}] ({source}, page {page})\n{doc.page_content}")
    return "\n\n".join(formatted)


if __name__ == "__main__":
    test_query = "punishment for murder"
    print(f"Query: {test_query}\n")
    chunks = retrieve_chunks(test_query, k=3)
    print(format_chunks_for_prompt(chunks))
