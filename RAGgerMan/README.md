# 📄 RAGgerMan: Conversational Document Q&A Assistant

RAGgerMan is a complete Conversational Retrieval-Augmented Generation (RAG) pipeline and Streamlit application built with LangChain, ChromaDB, Hugging Face Transformers, and Streamlit.

It enables you to upload documents (PDF, TXT, Markdown), index them into vector embeddings, and converse with a local language model that leverages conversation history to provide accurate, grounded answers.

---

## 🚀 Key Features

1. **Flexible Ingestion**: Supports local file paths and temporary uploaded files (Streamlit file uploader).
2. **Text Chunking**: Uses `RecursiveCharacterTextSplitter` with configurable chunk sizes and overlap.
3. **Embeddings & Vector Storage**: Dense embeddings using IBM Granite (`ibm-granite/granite-embedding-small-english-r2`) stored in a persistent ChromaDB vector store with cosine distance.
4. **History-Aware Question Reformulation**: Reformulates contextual follow-up questions into standalone queries before retrieval.
5. **MMR Retrieval**: Maximal Marginal Relevance (MMR) search for diverse, non-redundant context retrieval.
6. **Local LLM Inference**: Fully offline/local text generation using `SmolLM2-135M-Instruct` with `ChatHuggingFace`.
7. **Clean Streamlit UI**: User-friendly interface with multi-file uploader, source citation viewer, and database management.

---

## 🛠️ Project Structure

```
RAGgerMan/
├── full_rag_pipeline.py   # Complete RAG pipeline (Ingestion, Chunking, ChromaDB, History-Aware RAG)
├── app.py                 # Streamlit conversational web interface
├── requirements.txt       # Python dependencies
└── README.md              # Activation and usage guide
```

---

## 📦 How to Activate and Run the App

### 1. Activate your Python Virtual Environment
Open your terminal (PowerShell or Command Prompt) and activate your environment:

```powershell
# For the llmdev venv in this workspace:
c:\Users\abhay\code_llm_finetune\llmdev\Scripts\activate
```

### 2. Navigate to the RAGgerMan directory
```powershell
cd c:\Users\abhay\code_llm_finetune\raggerman\RAGgerMan
```

### 3. Install Required Dependencies (if not already installed)
```powershell
pip install -r requirements.txt
```

### 4. Launch the Streamlit App
```powershell
streamlit run app.py
```

The app will start and open automatically in your browser at `http://localhost:8501`.

---

## 💡 How to Use the App

1. **Upload Documents**: Use the file uploader in the sidebar to select your PDF, TXT, or Markdown documents.
2. **Ingest**: Click the **"Process & Ingest Documents"** button. The app will chunk the text, compute embeddings, and store them in ChromaDB.
3. **Ask Questions**: Type your question in the chat input bar at the bottom.
4. **Follow-up Questions**: Ask follow-up questions naturally (e.g. *"What are its advantages?"* or *"Can you summarize the second point?"*). The system uses the conversation history to understand context.
5. **Inspect Sources**: Expand the **"🔍 View Retrieved Sources"** accordion under any assistant response to view the exact text chunks used.
6. **Reset/Clear**: Use **"Clear Chat"** or **"Reset DB"** in the sidebar whenever you want to start a fresh session.
