import os
from langchain_community.document_loaders import PyPDFLoader as py
def get_pdf(data_folder="data/"):
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
            
            print(f"Successfully loaded: {file}")
            print(f"Total pages: {len(pages)}")
            print(f"First page preview: {pages[0].page_content[:100]}...\n")
            
        except Exception as e:
            print(f"Failed to load {file}: {str(e)}")

if __name__ == "__main__":
    get_pdf()

