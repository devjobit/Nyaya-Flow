import os
from langchain_community.document_loaders import PyPDFLoader as py
from langchain_text_splitters import RecursiveCharacterTextSplitter
def load_and_chunk_legal_docs(data_folder="data/"):
    all_chunks=[]
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150,length_function=len)
    if not os.path.exists(data_folder):
        print(f"Error: The folder {data_folder} is not found")
        return
    files=[f for f in os.listdir(data_folder) if f.endswith('.pdf')]
    if not files:
        print("No PDF files found in the data folder")
        return
    print(f"Found {len(files)} leagal documents")
    for file in files:
        file_path=os.path.join(data_folder, file)
        try:
            loader = py(file_path)
            pages = loader.load()
            for page in pages:
                page.metadata["law_source"] = file.replace(".pdf","")
            chunks=text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"{file}: Created {len(chunks)} chunks.")
            
        except Exception as e:
            print(f"Failed to load {file}: {str(e)}")
    return all_chunks

if __name__ == "__main__":
    final_chunks = load_and_chunk_legal_docs()
    if final_chunks:
        print(f"\n Total Chunks Ready: {len(final_chunks)}")
        print(f" Source of first chunk: {final_chunks[0].metadata['law_source']}")

