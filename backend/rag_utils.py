import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()


def load_n_split_documents(template_dir):
    # Manually load raw HTML content instead of using DirectoryLoader's .load() for text
    loaded_docs = []
    for root, _, files in os.walk(template_dir):
        for file in files:
            if file.endswith(".html"):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    loaded_docs.append(Document(page_content=content, metadata={'source': filepath}))
    print(f"Found {len(loaded_docs)} documents with raw HTML content.")
    # No splitting needed as we want the full HTML
    return loaded_docs

def create_vector_store(chunks):
    print(f"Creating vector store with {len(chunks)} chunks.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore