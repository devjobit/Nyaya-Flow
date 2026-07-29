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
│   ├── ingestion.py    # PDF ingestion, recursive chunking, and metadata tagging
│   ├── database.py     # Pinecone vector storage integration
│   ├── retrieval.py    # Queries Pinecone for relevant IPC/BNS chunks
│   ├── agent.py        # LangChain v1 agent (Groq LLM) that reasons over retrieved text
│   └── app.py          # Streamlit frontend
└── requirements.txt    # Dependency list
```

## ▶️ Running it

1. `pip install -r requirements.txt`
2. Fill in `.env` with real `PINECONE_API_KEY` and `GROQ_API_KEY` values
   (get a free Groq key at [console.groq.com](https://console.groq.com)).
3. Build the index once: `python -m src.database`
4. Launch the app: `streamlit run src/app.py`

You can also query the agent directly from the terminal: `python -m src.agent`
