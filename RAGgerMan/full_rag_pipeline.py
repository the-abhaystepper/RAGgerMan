import os
import sys
import tempfile
import torch
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

# Ensure UTF-8 output encoding for windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


"""
Function used to load in documents from either a file path (string) or an uploaded file object.
Handles temporary storage if an in-memory uploaded file is provided.
"""
def load_documents(file_input):
    #If the input is already a string path on disk
    if isinstance(file_input, str):
        if file_input.lower().endswith(".pdf"):
            loader = PyMuPDF4LLMLoader(
                file_path=file_input,
                mode="single",
                pages_delimiter="\n\f"
            )
            return loader.load()
        else:
            with open(file_input, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": file_input})]

    #If the input is a Streamlit uploaded file or byte stream
    filename = getattr(file_input, "name", "uploaded_document.pdf")
    suffix = os.path.splitext(filename)[1] if "." in filename else ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        if hasattr(file_input, "getbuffer"):
            temp_file.write(file_input.getbuffer())
        elif hasattr(file_input, "read"):
            file_input.seek(0)
            temp_file.write(file_input.read())
        elif isinstance(file_input, bytes):
            temp_file.write(file_input)
        temp_path = temp_file.name

    try:
        if suffix.lower() == ".pdf":
            loader = PyMuPDF4LLMLoader(
                file_path=temp_path,
                mode="single",
                pages_delimiter="\n\f"
            )
            documents = loader.load()
        else:
            with open(temp_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": filename})]

        #Update metadata to show clean source filename
        for doc in documents:
            doc.metadata["source"] = filename

        return documents
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


""" 
Function used to split the document text into chunks.
Instantiates the RecursiveCharacterTextSplitter class and returns the chunks.
"""
def split_documents(documents: list, chunk_size: int = 500, overlap: int = 50):
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    chunks = recursive_text_splitter.split_documents(documents)
    return chunks


"""
Function used to embed document chunks and add them to a Chroma vectorstore.
Instantiates HuggingFaceEmbeddings with the granite model and persists chunks in ChromaDB.
"""
def embed_and_store(chunks: list, store_directory: str = "db/ChromaDB"):
    model_id = "ibm-granite/granite-embedding-small-english-r2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=store_directory,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    if chunks:
        vectorstore.add_documents(chunks)
        
    return vectorstore


"""
Function used to load and configure the HuggingFace Chat Model.
"""
def load_chat_model(generation_model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(generation_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        generation_model_id,
        device_map="auto" if device == "cuda" else None,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.generation_config.max_new_tokens = 512
    model.generation_config.temperature = 0.2
    model.generation_config.do_sample = True
    model.generation_config.repetition_penalty = 1.1

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    chat_agent = ChatHuggingFace(llm=llm)
    return chat_agent


"""
Function to reformulate a follow-up user query into a standalone question based on conversation history.
"""
def create_history_aware_query(chat_agent, chat_history: list, user_query: str) -> str:
    if not chat_history:
        return user_query

    #Format recent chat history into readable conversation text
    history_text = ""
    for msg in chat_history[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    reformulate_prompt = f"""Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that can be understood on its own without the conversation history. Do NOT answer the question, only return the rewritten question.

Conversation History:
{history_text}

Follow-up Question: {user_query}

Standalone Question:"""

    response = chat_agent.invoke([HumanMessage(content=reformulate_prompt)])
    standalone_query = response.content.strip()
    return standalone_query if standalone_query else user_query


"""
Main RAG generation function with history-aware retrieval and grounded generation.
"""
def rag_generate(chat_agent, retriever, user_query: str, chat_history: list = None):
    if chat_history is None:
        chat_history = []

    #History-aware query reformulation if conversation exists
    standalone_query = create_history_aware_query(chat_agent, chat_history, user_query)

    #Retrieve documents using standalone query with MMR search
    retrieved_docs = retriever.invoke(standalone_query)

    #Format combined input with retrieved context
    context_text = "\n".join([f"- (Source: {doc.metadata.get('source', 'Unknown')}) {doc.page_content}" for doc in retrieved_docs])
    
    combined_input = f"""Based on the following documents, please answer this question: {user_query}

Documents:
{context_text}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    messages = [
        SystemMessage(content="You are a helpful assistant for document question answering.")
    ]
    
    #Add previous chat history for conversational awareness
    for msg in chat_history[-6:]:
        messages.append(msg)
        
    messages.append(HumanMessage(content=combined_input))

    #Invoke the chat model and return answer and source documents
    ai_message = chat_agent.invoke(messages)
    
    return {
        "answer": ai_message.content.strip(),
        "standalone_query": standalone_query,
        "retrieved_docs": retrieved_docs
    }
