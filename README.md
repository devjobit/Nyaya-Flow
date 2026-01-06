# Nyaya-Flow: AI Legal Intelligence for the Indian Transition ⚖️🤖

Nyaya-Flow is an Agentic RAG (Retrieval-Augmented Generation) system designed to help legal professionals navigate the transition from the **Indian Penal Code (IPC)** to the new **Bhartiya Nyaya Sanhita (BNS)**.

## 🚀 The Problem
India is replacing century-old laws. Lawyers and law students need a way to instantly map old sections to new ones and understand the nuances of the 2023 reforms. Nyaya-Flow uses AI to bridge this gap.

## 🛠 Tech Stack
- **Language:** Python 3.10+
- **AI Framework:** LangChain / LlamaIndex
- **Vector Database:** Pinecone (Serverless)
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **LLM:** Llama 3.1 (via Groq) / Gemini 1.5 Flash
- **UI:** Streamlit

## 📁 Project Structure
```text
Nyaya-Flow/
├── data/               # Official Gazette PDFs of BNS & IPC
├── src/                
│   ├── ingestion.py    # Document processing & chunking logic
│   ├── database.py     # Pinecone vector storage integration
│   └── app.py          # Streamlit frontend (Coming Soon)
└── requirements.txt    # Dependency list
