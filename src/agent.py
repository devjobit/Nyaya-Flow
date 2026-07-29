"""
agent.py
The "Agentic" layer of Nyaya-Flow. Wraps the retrieval layer as LangChain tools
and lets a Groq-hosted LLM decide when/how to search the IPC and BNS statutes,
then synthesizes a grounded answer (e.g. mapping an IPC section to its BNS
equivalent, or explaining what changed).

Built on LangChain 1.x's `create_agent` (the langgraph-based agent runtime).
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

from src.retrieval import retrieve_chunks, format_chunks_for_prompt

load_dotenv()

IPC_SOURCE = "Indian Penal Code 1860 official PDF"
BNS_SOURCE = "Bhartiya Nyaya Sanhita 2023 PDF"

# Groq deprecated llama-3.3-70b-versatile in mid-2026; gpt-oss-120b is the
# recommended, currently-supported replacement. Override via .env if needed.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "groq:openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are Nyaya-Flow, an AI legal research assistant specializing in India's \
transition from the Indian Penal Code (IPC, 1860) to the Bhartiya Nyaya Sanhita (BNS, 2023).

Rules you must follow:
1. Always ground your answer in the text returned by your tools. Never invent section numbers \
or punishments from memory alone — the IPC-to-BNS renumbering is not a simple offset and \
guessing is dangerous for a legal-research tool.
2. When asked to map an IPC section to BNS (or vice versa), search BOTH statutes and clearly \
state: the original section, the corresponding new section, and what (if anything) materially \
changed in wording or punishment.
3. If the retrieved context doesn't contain enough information to answer confidently, say so \
plainly instead of guessing.
4. This tool is for legal research and education, not a substitute for a licensed advocate. \
For anything touching an active case, tell the user to confirm with a lawyer.
5. Be concise and structured — use short sections or bullet points, not dense paragraphs.
"""


@tool
def search_ipc(query: str) -> str:
    """Search the Indian Penal Code (IPC, 1860) for sections relevant to the query.
    Use this to find the ORIGINAL/OLD law text (e.g. 'Section 302 murder')."""
    chunks = retrieve_chunks(query, k=4, law_source=IPC_SOURCE)
    return format_chunks_for_prompt(chunks)


@tool
def search_bns(query: str) -> str:
    """Search the Bhartiya Nyaya Sanhita (BNS, 2023) for sections relevant to the query.
    Use this to find the NEW law text that replaced the IPC."""
    chunks = retrieve_chunks(query, k=4, law_source=BNS_SOURCE)
    return format_chunks_for_prompt(chunks)


@tool
def search_both(query: str) -> str:
    """Search across both IPC and BNS without filtering. Useful for a broad first pass
    when you don't yet know which statute the answer lives in."""
    chunks = retrieve_chunks(query, k=6)
    return format_chunks_for_prompt(chunks)


def build_agent(model: str = DEFAULT_MODEL):
    """Constructs the Nyaya-Flow agent graph."""
    return create_agent(
        model=model,
        tools=[search_ipc, search_bns, search_both],
        system_prompt=SYSTEM_PROMPT,
    )


def ask(question: str, model: str = DEFAULT_MODEL) -> str:
    """Runs a single question through the agent and returns the final text answer."""
    agent = build_agent(model=model)
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    demo_question = "What is the BNS equivalent of IPC Section 302 (murder), and what changed?"
    print(f"Q: {demo_question}\n")
    print(ask(demo_question))
