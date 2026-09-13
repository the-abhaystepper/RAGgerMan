import sys

#ensure UTF-8 output encoding for windows terminal (doesn't work otherwise)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pyarrow
import torch
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.messages import HumanMessage, SystemMessage


#Instantiating the persistant paths, embedding model and ChromaDB
store_dir = "db/chroma_db"

model_id = "ibm-granite/granite-embedding-small-english-r2"
embedding_model = HuggingFaceEmbeddings(
    model_name=model_id,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

db = Chroma(
    persist_directory=store_dir,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)


#Instantiating the ChromaDB retriever to perform the similarity search for the user query
user_query = "From where did OpenAI get its dgx-1 supercomputer?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3
    }
)

retrieved_docs = retriever.invoke(user_query)


#Loading in the Chat Model for text generation
generation_model_id = "google/gemma-3-270m-it"

llm = HuggingFacePipeline.from_model_id(
    model_id=generation_model_id,
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
    },
    model_kwargs={
        "device_map": "auto",
    }
)

chat_agent = ChatHuggingFace(llm=llm)


#Formatting the final input to the Chat Model (combined query and retrieved documents)
combined_input = f"""Based on the following documents, please answer this question: {user_query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in retrieved_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]


#Invoking the Chat Model with the combined input and outputting response
ai_message = chat_agent.invoke(messages)

print(ai_message.content)