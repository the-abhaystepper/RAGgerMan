import mimetypes
import os
import sys

# Windows may serve .js as text/plain, which blocks Streamlit widget chunks.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

#Ensure python can import from current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from full_rag_pipeline import (
    load_documents,
    split_documents,
)
from rag_service import RagService

#Page configuration
st.set_page_config(
    page_title="RAGgerMan",
    layout="wide"
)

st.title("RAGgerMan: RAG pipeline for SLMs")
st.caption("Upload documents, index them with ChromaDB and Granite Embeddings, and chat using locally run SLMs.")


#Cache models so they don't reload on every UI interaction.
#CUDA models cannot be loaded on Streamlit's script thread on Windows
#(access violation after "Loading weights"), so they run in a child process.
@st.cache_resource
def get_rag_service():
    return RagService(store_dir=os.path.join(current_dir, "db", "ChromaDB"))


#Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []


#Sidebar for document ingestion and database controls
with st.sidebar:
    st.header("Document Ingestion")
    
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or Markdown documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    ingest_clicked = st.button("Process & Ingest Documents", type="primary", use_container_width=True)
    st.divider()
    st.subheader("Indexed Knowledge Base")
    count_slot = st.empty()
    files_slot = st.empty()
    st.divider()

    #Clear options
    col1, col2 = st.columns(2)
    with col1:
        clear_chat = st.button("Clear Chat", use_container_width=True)
    with col2:
        reset_db = st.button("Reset DB", use_container_width=True)

try:
    service = get_rag_service()
except Exception as exc:
    st.error("Could not start the RAG worker process.")
    st.exception(exc)
    st.stop()

if ingest_clicked:
    if uploaded_files:
        total_chunks = 0
        with st.spinner("Processing and indexing documents into ChromaDB"):
            for uploaded_file in uploaded_files:
                #Load document (handles temp storage automatically)
                docs = load_documents(uploaded_file)
                
                #plit into chunks
                chunks = split_documents(docs, chunk_size=500, overlap=50)
                
                #Embed and add to vectorstore
                if chunks:
                    total_chunks += service.add_documents(chunks)
                    if uploaded_file.name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(uploaded_file.name)
        
        st.success(f"Successfully ingested {len(uploaded_files)} file(s) ({total_chunks} chunks).")
    else:
        st.warning("Please select at least one file to upload.")

if clear_chat:
    st.session_state.messages = []
    st.session_state.langchain_history = []
    st.rerun()

if reset_db:
    try:
        service.reset()
    except Exception:
        pass
    st.session_state.uploaded_files_list = []
    st.session_state.messages = []
    st.session_state.langchain_history = []
    st.success("Vector DB reset.")
    st.rerun()

try:
    count = service.chunk_count()
except Exception:
    count = 0
count_slot.write(f"**Total chunks in DB:** {count}")
if st.session_state.uploaded_files_list:
    file_lines = "\n".join(f"- `{fname}`" for fname in st.session_state.uploaded_files_list)
    files_slot.markdown(f"**Loaded Files:**\n{file_lines}")


#Main Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View retrieved sources"):
                for i, doc in enumerate(msg["sources"], 1):
                    metadata = doc["metadata"] if isinstance(doc, dict) else doc.metadata
                    page_content = doc["page_content"] if isinstance(doc, dict) else doc.page_content
                    st.markdown(f"**Source {i}:** `{metadata.get('source', 'Unknown')}`")
                    st.text(page_content.strip())


#Input Box
user_query = st.chat_input("Ask a question about your documents")

if user_query:
    #Display user query in chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if count == 0:
        warning_reply = "No documents found in the database. Please upload and ingest a document first using the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": warning_reply})
        with st.chat_message("assistant"):
            st.warning(warning_reply)
    else:
        #Run RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating answer"):
                history = []
                for item in st.session_state.langchain_history:
                    role = "user" if isinstance(item, HumanMessage) else "assistant"
                    history.append({"role": role, "content": item.content})

                result = service.chat(user_query=user_query, chat_history=history)
                answer = result["answer"]
                retrieved_docs = result["retrieved_docs"]

                st.markdown(answer)

                if retrieved_docs:
                    with st.expander("View retrieved sources"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.markdown(f"**Source {i}:** `{doc['metadata'].get('source', 'Unknown')}`")
                            st.text(doc["page_content"].strip())

        #Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved_docs
        })
        st.session_state.langchain_history.append(HumanMessage(content=user_query))
        st.session_state.langchain_history.append(AIMessage(content=answer))
