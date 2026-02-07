import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.ingestion import load_and_chunk_legal_docs

# Load secrets from .env
load_dotenv()

def sync_to_pinecone(index_name="nyaya-flow"):
    """Turns text into vectors and uploads them to Pinecone."""
    
    # 1. Get the data from our ingestion script
    chunks = load_and_chunk_legal_docs()
    if not chunks:
        print("⚠️ No chunks found to upload.")
        return

    # 2. Initialize the 'Translator' (Embedding Model)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Connect and Upload to Pinecone
    print(f"📡 Connecting to Pinecone index: {index_name}...")
    vectorstore = PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=index_name
    )
    
    print("✨ Successfully uploaded all legal chunks to the cloud!")
    return vectorstore

if __name__ == "__main__":
    sync_to_pinecone()
