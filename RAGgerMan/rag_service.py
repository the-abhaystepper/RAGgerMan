import argparse
import json
import os
import socket
import struct
import sys
import threading
import traceback

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


def _recv_msg(sock):
    header = b""
    while len(header) < 4:
        chunk = sock.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("Worker socket closed.")
        header += chunk
    size = struct.unpack(">I", header)[0]
    payload = b""
    while len(payload) < size:
        chunk = sock.recv(size - len(payload))
        if not chunk:
            raise ConnectionError("Worker socket closed.")
        payload += chunk
    return json.loads(payload.decode("utf-8"))


def _send_msg(sock, obj):
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _run_worker(store_dir: str) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    sys.stdout.write(f"RAGGERMAN_PORT {port}\n")
    sys.stdout.flush()
    # Keep the port handshake on the pipe; send model logs to the terminal.
    sys.stdout = sys.stderr

    from full_rag_pipeline import embed_and_store, load_chat_model, rag_generate

    vectorstore = embed_and_store(chunks=[], store_directory=store_dir)
    chat_agent = None

    conn, _addr = server.accept()
    try:
        while True:
            request = _recv_msg(conn)
            method = request.get("method")
            try:
                if method == "count":
                    try:
                        result = int(vectorstore._collection.count())
                    except Exception:
                        result = 0
                elif method == "add_documents":
                    docs = [
                        Document(page_content=item["page_content"], metadata=item.get("metadata") or {})
                        for item in request.get("documents") or []
                    ]
                    if docs:
                        vectorstore.add_documents(docs)
                    result = len(docs)
                elif method == "reset":
                    try:
                        vectorstore.delete_collection()
                    except Exception:
                        pass
                    vectorstore = embed_and_store(chunks=[], store_directory=store_dir)
                    result = True
                elif method == "chat":
                    if chat_agent is None:
                        chat_agent = load_chat_model()
                    history = []
                    for item in request.get("chat_history") or []:
                        if item["role"] == "user":
                            history.append(HumanMessage(content=item["content"]))
                        else:
                            history.append(AIMessage(content=item["content"]))
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5},
                    )
                    rag_result = rag_generate(
                        chat_agent=chat_agent,
                        retriever=retriever,
                        user_query=request["user_query"],
                        chat_history=history,
                    )
                    result = {
                        "answer": rag_result["answer"],
                        "retrieved_docs": [
                            {
                                "page_content": doc.page_content,
                                "metadata": dict(doc.metadata or {}),
                            }
                            for doc in rag_result["retrieved_docs"]
                        ],
                    }
                elif method == "shutdown":
                    _send_msg(conn, {"ok": True, "result": True})
                    break
                else:
                    raise ValueError(f"Unknown method: {method}")
                _send_msg(conn, {"ok": True, "result": result})
            except Exception:
                _send_msg(conn, {"ok": False, "error": traceback.format_exc()})
    finally:
        conn.close()
        server.close()


class RagService:
    def __init__(self, store_dir: str = "db/ChromaDB", ready_timeout: float = 180.0):
        import subprocess

        self._lock = threading.Lock()
        worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_service.py")
        self._proc = subprocess.Popen(
            [sys.executable, worker_path, "--store-dir", os.path.abspath(store_dir)],
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
            bufsize=1,
        )
        port = None
        deadline_chunks = 0
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("RAG worker exited before publishing its port.")
            line = line.strip()
            if line.startswith("RAGGERMAN_PORT "):
                port = int(line.split()[1])
                break
            deadline_chunks += 1
            if deadline_chunks > 10000:
                raise RuntimeError("Timed out waiting for RAG worker port.")

        self._sock = socket.create_connection(("127.0.0.1", port), timeout=ready_timeout)
        self._sock.settimeout(300)

    def _call(self, payload: dict):
        if self._proc.poll() is not None:
            raise RuntimeError("RAG worker process is not running.")
        with self._lock:
            _send_msg(self._sock, payload)
            response = _recv_msg(self._sock)
        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "RAG worker request failed.")
        return response["result"]

    def chunk_count(self) -> int:
        return int(self._call({"method": "count"}))

    def add_documents(self, documents: list[Document]) -> int:
        payload = [
            {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}
            for doc in documents
        ]
        return int(self._call({"method": "add_documents", "documents": payload}))

    def reset(self) -> None:
        self._call({"method": "reset"})

    def chat(self, user_query: str, chat_history: list[dict]) -> dict:
        return self._call({"method": "chat", "user_query": user_query, "chat_history": chat_history})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", default="db/ChromaDB")
    args = parser.parse_args()
    _run_worker(args.store_dir)
