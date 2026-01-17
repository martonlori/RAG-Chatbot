import os
from pathlib import Path
import faiss
import numpy as np
import ollama

# Introducing constants for data directory, chunking, and target data
DATA_DIR = Path("data/plain_text_articles")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
INDEX_PATH = Path("data/vector_store/faiss.index")
META_PATH = Path("data/vector_store/metadata.npy")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def load_and_chunk_all_files():
    all_chunks = []
    metadatas = []

    for file_path in DATA_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadatas.append({
                "source": file_path.name,
                "chunk_id": i,
                "text": chunk
            })

    return all_chunks, metadatas


def embed_texts(texts, model="nomic-embed-text"):
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=model, prompt=text)
        embeddings.append(response["embedding"])
    return np.array(embeddings).astype("float32")


def main():
    print("Loading and chunking documents...")
    chunks, metadatas = load_and_chunk_all_files()
    print(f"Loaded {len(chunks)} chunks")

    print("Generating embeddings with Ollama...")
    embeddings = embed_texts(chunks)
    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("Saving index and metadata...")
    faiss.write_index(index, str(INDEX_PATH))
    np.save(META_PATH, metadatas)

    print("Done")


if __name__ == "__main__":
    main()