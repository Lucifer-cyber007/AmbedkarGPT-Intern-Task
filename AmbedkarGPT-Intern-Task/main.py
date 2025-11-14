"""
main.py — Clean AmbedkarGPT (silent fallback, no warning prints)

Behavior:
- Uses Chroma + HuggingFace embeddings when available (RAG).
- Silently falls back to full-text Ollama if retrieval returns no docs.
- Does NOT print the "warning" or "no vectordb available" messages.
- Shows only the final answer and retrieved snippets (if any).
"""

import os
import sys
import subprocess
import importlib
from typing import Any, List, Optional

# Silence external deprecation/warning noise
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Robust import helper
# -------------------------
def try_import(module_name: str, attr: Optional[str] = None):
    try:
        m = importlib.import_module(module_name)
        return getattr(m, attr) if attr else m
    except Exception:
        return None

# modular imports (pick whichever exists)
TextLoader = try_import("langchain_community.document_loaders", "TextLoader") or try_import("langchain.document_loaders", "TextLoader")
CharacterTextSplitter = try_import("langchain_text_splitters", "CharacterTextSplitter") or try_import("langchain.text_splitters", "CharacterTextSplitter")
HuggingFaceEmbeddings = try_import("langchain.embeddings", "HuggingFaceEmbeddings") or try_import("langchain_huggingface", "HuggingFaceEmbeddings")
Chroma = try_import("langchain_community.vectorstores", "Chroma") or try_import("langchain.vectorstores", "Chroma") or try_import("chromadb", "Client")
OllamaLLM = try_import("langchain_ollama", "Ollama") or try_import("langchain.llms", "Ollama")
RetrievalQA = try_import("langchain_community.chains", "RetrievalQA") or try_import("langchain.chains", "RetrievalQA")
DocumentClass = try_import("langchain_core.documents", "Document") or try_import("langchain.schema", "Document")

# -------------------------
# Config
# -------------------------
SPEECH_FILE = "speech.txt"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
OLLAMA_MODEL = "mistral"  # change to smaller model if needed
OLLAMA_TIMEOUT = 180      # seconds per attempt for ollama subprocess

# -------------------------
# Ollama subprocess (robust)
# -------------------------
def run_ollama_subprocess(model: str, prompt: str, timeout: int = OLLAMA_TIMEOUT) -> str:
    attempts = 2
    backoff = 2
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            proc = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
            )
            if proc.returncode == 0 and proc.stdout and proc.stdout.strip():
                return proc.stdout.strip()
            last_err = (proc.returncode, (proc.stderr or "").strip(), (proc.stdout or "").strip())
        except FileNotFoundError:
            raise RuntimeError("ollama binary not found. Install Ollama and ensure 'ollama' is in PATH.")
        except subprocess.TimeoutExpired:
            last_err = f"Timeout after {timeout}s"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

        if attempt < attempts:
            import time
            time.sleep(backoff ** attempt)

    raise RuntimeError(f"Ollama calls failed. Last error: {last_err}")

# -------------------------
# Vectorstore build/load
# -------------------------
def ensure_speech_file():
    if not os.path.isfile(SPEECH_FILE):
        print(f"ERROR: {SPEECH_FILE} not found in {os.getcwd()}")
        sys.exit(1)

def build_or_load_vectorstore():
    ensure_speech_file()

    if Chroma is None or HuggingFaceEmbeddings is None:
        return None

    try:
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception:
        return None

    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        try:
            vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
            return vectordb
        except Exception:
            pass

    # load file
    docs = None
    if TextLoader:
        try:
            loader = TextLoader(SPEECH_FILE, encoding="utf-8")
            docs = loader.load()
        except Exception:
            docs = None
    if docs is None:
        with open(SPEECH_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        docs = [type("D", (), {"page_content": text})()]

    # split
    if CharacterTextSplitter:
        try:
            splitter = CharacterTextSplitter(separator="\n", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            split_docs = splitter.split_documents(docs)
        except Exception:
            raw = docs[0].page_content if hasattr(docs[0], "page_content") else str(docs[0])
            split_docs = [type("D", (), {"page_content": raw[i:i+CHUNK_SIZE]})() for i in range(0, len(raw), CHUNK_SIZE-CHUNK_OVERLAP)]
    else:
        raw = docs[0].page_content if hasattr(docs[0], "page_content") else str(docs[0])
        parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
        split_docs = [type("D", (), {"page_content": p})() for p in parts]

    # create DB
    try:
        vectordb = Chroma.from_documents(split_docs, embedding, persist_directory=PERSIST_DIR)
        try:
            vectordb.persist()
        except Exception:
            pass
        return vectordb
    except Exception:
        return None

# -------------------------
# Retrieval helper: try many APIs, return list
# -------------------------
def get_documents_from_retriever(obj, question: str, k: int = TOP_K) -> List[Any]:
    if obj is None:
        return []

    # try common retriever API
    try:
        if hasattr(obj, "get_relevant_documents"):
            docs = obj.get_relevant_documents(question)
            return docs or []
    except Exception:
        pass

    # alternative text-based
    try:
        if hasattr(obj, "get_relevant_texts"):
            texts = obj.get_relevant_texts(question)
            return [type("D", (), {"page_content": t})() for t in (texts or [])]
    except Exception:
        pass

    # vectorstore methods
    try:
        if hasattr(obj, "similarity_search"):
            return obj.similarity_search(question, k=k) or []
    except Exception:
        pass

    try:
        if hasattr(obj, "similarity_search_with_score"):
            results = obj.similarity_search_with_score(question, k=k) or []
            docs = []
            for it in results:
                if isinstance(it, tuple):
                    docs.append(it[0])
                else:
                    docs.append(it)
            return docs
    except Exception:
        pass

    try:
        if hasattr(obj, "search"):
            return obj.search(question, k=k) or []
    except Exception:
        pass

    try:
        if hasattr(obj, "retrieve"):
            return obj.retrieve(question, k=k) or []
    except Exception:
        pass

    # if obj is a vectorstore, try as_retriever
    try:
        if hasattr(obj, "as_retriever"):
            r = obj.as_retriever(search_kwargs={"k": k})
            if hasattr(r, "get_relevant_documents"):
                return r.get_relevant_documents(question) or []
            if hasattr(r, "similarity_search"):
                return r.similarity_search(question, k=k) or []
    except Exception:
        pass

    return []

# -------------------------
# create_qa_chain: return chain / retriever / fallback callable
# -------------------------
def create_qa_chain(vectordb):
    # try to create LangChain RetrievalQA if possible
    try:
        if RetrievalQA is not None and OllamaLLM is not None and vectordb is not None:
            llm = OllamaLLM(model=OLLAMA_MODEL)
            retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
            return qa_chain
    except Exception:
        pass

    # if vectordb exists, prefer returning its retriever (or vectordb itself)
    if vectordb is not None:
        try:
            r = vectordb.as_retriever(search_kwargs={"k": TOP_K})
            return r
        except Exception:
            return vectordb

    # final fallback: callable that queries full-text
    def full_text_fallback(question: str):
        try:
            with open(SPEECH_FILE, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            return {"answer": f"Failed to read text: {e}", "source_documents": []}

        prompt = (
            "You are AmbedkarGPT. Use ONLY the provided text below to answer the question.\n"
            "If the answer cannot be found in the text, reply exactly: Insufficient information.\n\n"
            "TEXT:\n" + text + "\n\nQUESTION:\n" + question + "\n\nAnswer concisely:"
        )
        try:
            out = run_ollama_subprocess(OLLAMA_MODEL, prompt)
            return {"answer": out, "source_documents": []}
        except Exception as e:
            return {"answer": f"Failed to call Ollama: {e}", "source_documents": []}

    return full_text_fallback

# -------------------------
# Clean CLI - NO warning prints
# -------------------------
def cli_loop(qa_chain):
    print("\nAmbedkarGPT — Ask anything from the speech. (type 'exit' to quit)")

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        result = None

        # 1) If chain is callable try common call patterns
        if callable(qa_chain):
            try:
                if hasattr(qa_chain, "run"):
                    out = qa_chain.run(question)
                    result = {"answer": out} if isinstance(out, str) else out
            except Exception:
                result = None

            if result is None:
                try:
                    if hasattr(qa_chain, "predict"):
                        out = qa_chain.predict(question)
                        result = {"answer": out} if isinstance(out, str) else out
                except Exception:
                    result = None

            if result is None:
                try:
                    out = qa_chain(question)
                    result = {"answer": out} if isinstance(out, str) else out
                except Exception:
                    result = None

        # 2) If no result yet, attempt retrieval silently
        if result is None:
            docs = get_documents_from_retriever(qa_chain, question, k=TOP_K)

            if docs:
                sources_text = ""
                for d in docs:
                    content = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
                    sources_text += f"---\n{content}\n"

                prompt = (
                    "You are AmbedkarGPT. Use ONLY the provided source snippets.\n"
                    "If the answer cannot be found, reply: Insufficient information.\n\n"
                    "SNIPPETS:\n"
                    f"{sources_text}\nQUESTION:\n{question}\n\nAnswer concisely:"
                )
                try:
                    answer_text = run_ollama_subprocess(OLLAMA_MODEL, prompt)
                    result = {"answer": answer_text, "source_documents": docs}
                except Exception as e:
                    # fallback to full text silently
                    qa_fallback = create_qa_chain(None)
                    result = qa_fallback(question)
            else:
                # silent full-text fallback
                qa_fallback = create_qa_chain(None)
                result = qa_fallback(question)

        # Normalize and print
        if isinstance(result, dict):
            answer = result.get("answer") or result.get("result") or result.get("output_text") or ""
            sources = result.get("source_documents", []) or []
        else:
            answer = str(result)
            sources = []

        print("\nAnswer:\n", answer.strip(), "\n")
        if sources:
            print("Source snippets:")
            for i, doc in enumerate(sources, start=1):
                snippet = getattr(doc, "page_content", None) or getattr(doc, "content", None) or str(doc)
                print(f"--- Source {i} ---")
                print(snippet[:500].strip(), "...\n")

# -------------------------
# main
# -------------------------
def main():
    ensure_speech_file()
    vectordb = build_or_load_vectorstore()
    qa_chain = create_qa_chain(vectordb)
    cli_loop(qa_chain)

if __name__ == "__main__":
    main()
