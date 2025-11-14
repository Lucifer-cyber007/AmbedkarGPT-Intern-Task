# AmbedkarGPT — RAG Q&A (Ollama · **llama3.2:1b**)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Model](https://img.shields.io/badge/ollama-llama3.2:1b-lightgrey)](https://ollama.com/)
[![ChromaDB](https://img.shields.io/badge/vectorstore-Chroma-orange)](https://github.com/chroma-core/chroma)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/<GITHUB_USER>/<REPO>/actions/workflows/ci.yml/badge.svg)](https://github.com/<GITHUB_USER>/<REPO>/actions/workflows/ci.yml) <!-- replace placeholders -->

---

<!-- Animated demo GIF (add demo.gif to repo root) -->
<p align="center">
  <a href="https://github.com/<GITHUB_USER>/<REPO>">
    <img src="demo.gif" alt="AmbedkarGPT demo" width="800" />
  </a>
</p>

---

## What is AmbedkarGPT?

AmbedkarGPT is a local Retrieval-Augmented Generation (RAG) system that answers user questions based on Dr. B.R. Ambedkar’s speech.  
It uses modular LangChain packages, HuggingFace sentence-transformers for embeddings, ChromaDB for vector persistence, and **Ollama** with **llama3.2:1b** for local generation.

---

## Key features

- ✅ Local RAG pipeline (no cloud APIs)
- ✅ HuggingFace embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- ✅ ChromaDB persistent vector store
- ✅ Ollama inference using `llama3.2:1b` (fast on laptops)
- ✅ Robust fallbacks and retries on Windows
- ✅ Clean CLI that shows answers and retrieved snippets

---

## Quickstart (Windows / PowerShell)

```powershell
# 1. Clone
git clone https://github.com/<GITHUB_USER>/<REPO>.git
cd <REPO>

# 2. Create & activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama and pull model
# Install Ollama from https://ollama.com/download
ollama pull llama3.2:1b

# 5. Ensure speech.txt exists (project root)
# 6. Run
python main.py
