# RAG Chatbot

This project is a minimal **Retrieval-Augmented Generation (RAG)** chatbot built in Python.  
Its goal is to demonstrate how Large Language Models (LLMs) can be combined with vector search to answer questions based on a private document knowledge base.

The chatbot uses **SAP Help Articles** as its knowledge source and allows users to ask natural language questions, receiving answers grounded in the retrieved documentation context.

---

## What This Project Does

1. Takes a collection of cleaned `.txt` documents
2. Splits them into overlapping text chunks
3. Generates vector embeddings using a local embedding model via **Ollama**
4. Stores embeddings in a **FAISS vector database**
5. At query time:
   - Embeds the user question
   - Retrieves the most relevant chunks
   - Injects them into a prompt
   - Generates a final answer using a local LLM

This implements a full **RAG pipeline** end-to-end.

---

## Key Concepts Used

- Retrieval-Augmented Generation (RAG)
- Vector embeddings
- Similarity search
- Chunking and metadata mapping
- Prompt grounding
- Local LLM inference

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python** | Core programming language |
| **Ollama** | Local LLM and embedding inference |
| **FAISS** | Vector similarity search engine |
| **NumPy** | Numerical vector storage and manipulation |
| **PDF parsing tools** | Extracting and cleaning documents |

---

## Data

The knowledge base consists of:
- SAP Help Articles
- Extracted from PDFs
- Cleaned and converted into plain text
- Chunked into overlapping segments for semantic retrieval

Each chunk is stored with metadata linking it back to its source document.

---

## How the RAG Pipeline Works

1. **Offline (Indexing stage)**:
   - Documents → chunks → embeddings → FAISS index + metadata

2. **Online (Query stage)**:
   - User query → embedding → FAISS similarity search → top-k chunks
   - Retrieved chunks + user question → prompt → LLM → final answer
