"""
app.py
Streamlit frontend for Nyaya-Flow. Lets a user ask legal questions about the
IPC-to-BNS transition and see grounded, cited answers from the agentic RAG
pipeline in agent.py.

Run with:  streamlit run src/app.py
"""

import streamlit as st

from src.agent import ask

st.set_page_config(
    page_title="Nyaya-Flow | IPC ↔ BNS Legal Intelligence",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ Nyaya-Flow")
st.caption("Agentic RAG assistant for navigating India's IPC → BNS legal transition")

with st.sidebar:
    st.header("About")
    st.write(
        "Nyaya-Flow retrieves the actual text of the Indian Penal Code (1860) "
        "and the Bhartiya Nyaya Sanhita (2023) from a Pinecone vector store, "
        "then uses an LLM agent to map sections, compare wording, and explain "
        "what changed."
    )
    st.warning(
        "⚠️ Educational / research tool only. Not a substitute for advice from "
        "a licensed advocate, especially for active cases.",
        icon="⚠️",
    )
    st.divider()
    st.subheader("Try asking")
    st.markdown(
        "- What is the BNS equivalent of IPC Section 302?\n"
        "- How has the punishment for theft changed?\n"
        "- Summarize the key differences for sedition-related offences.\n"
        "- Which BNS section corresponds to IPC Section 420 (cheating)?"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_question = st.chat_input("Ask about an IPC section, a BNS section, or how one maps to the other...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching IPC & BNS statutes..."):
            try:
                answer = ask(user_question)
            except EnvironmentError as e:
                answer = (
                    f"⚠️ Configuration error: {e}\n\n"
                    "Make sure `PINECONE_API_KEY` and `GROQ_API_KEY` are set in your `.env` file, "
                    "and that you've run `python -m src.database` at least once to populate Pinecone."
                )
            except Exception as e:
                answer = f"⚠️ Something went wrong while answering: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
