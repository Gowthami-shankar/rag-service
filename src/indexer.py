# src/indexer.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

def build_index(crawled_data: dict, chunk_size: int, chunk_overlap: int, index_path: str):
    """
    Builds a vector index from the crawled text content.
    """
    documents = []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for url, text in crawled_data.items():
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(page_content=chunk, metadata={"source": url})
            documents.append(doc)

    if not documents:
        return 0
        
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    print(f"Creating FAISS index with {len(documents)} document chunks...")
    vector_store = FAISS.from_documents(documents, embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    vector_store.save_local(index_path)
    
    return len(documents)