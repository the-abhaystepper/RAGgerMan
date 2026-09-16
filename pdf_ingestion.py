from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_chroma import Chroma


loader = PyMuPDF4LLMLoader(
    file_path="docs/pdfs/attention-is-all-you-need.pdf",
    mode="single",
    pages_delimiter="\n\f"
)

documents = loader.load()
