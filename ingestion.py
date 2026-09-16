from jinja2 import loaders
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


"""
Function used to load in the documents from a local folder "./docs"
Instantiates the DirectoryLoader class, invokes the load method, and then returns the loaded documents
"""
def load_documents(doc_path):
    loader = PyMuPDF4LLMLoader(
        file_path="docs/pdfs/attention-is-all-you-need.pdf",
        mode="single",
        pages_delimiter="\n\f"
    )

    documents = loader.load()
    return documents

""" 
Function used to split the document text into chunks
Instantiates the CharacterTextSplitter class, invokes the split_documents method, and then returns the chunks
"""
def split_documents(documents: list, chunk_size: int, overlap: int):

    recursive_text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks = recursive_text_splitter.split_documents(documents)

    return chunks

"""
Function used to embed the document chunks and add them to a vectorstore
Instantiates the HuggingFaceEmbeddings class with embedding model, then uses the Chroma.from_documents method to embed and store the chunks in the ChromaDB vector database
"""
def embed_and_store(chunks: list, store_directory: str):

    model_id = "ibm-granite/granite-embedding-small-english-r2"
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=store_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vectorstore


def main():
    #Instantiating the persistant paths, embedding model and ChromaDB
    model_id = "ibm-granite/granite-embedding-small-english-r2"
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    store_dir = "db/ChromaDB"
    document_path = "docs/pdfs"
    vectorstore = Chroma(
        persist_directory= store_dir,
        embedding_function= embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    #Full ingestion pipeline
    documents = load_documents(doc_path=document_path)

    chunks = split_documents(documents=documents, chunk_size=500, overlap=0)

    vectorstore = embed_and_store(chunks=chunks, store_directory=store_dir)
    return vectorstore
    
    

if __name__ == "__main__":
    main()
    