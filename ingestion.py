import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_chroma import Chroma


def load_documents(doc_path="docs"):

    loader = DirectoryLoader(
        doc_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )

    documents = loader.load()
    print(len(documents))

    return documents


def split_documents(documents: list, chunk_size: int, overlap: int):

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def embed_and_store(chunks: list, model: str, store_directory: str):

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "ibm-granite/granite-embedding-small-english-r2"
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

    store_dir = "db/ChromaDB"
    document_path = "docs"
    vectorstore = Chroma(
        persist_directory= store_dir,
        embedding_function= embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )


    documents = load_documents(doc_path=document_path)

    chunks = split_documents(documents=documents, chunk_size=800, overlap=0)

    vectorstore = embed_and_store(chunks=chunks, model=embedding_model, store_directory=store_dir)
    return vectorstore
    
    

if __name__ == "__main__":
    main()
    