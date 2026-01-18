import faiss
import numpy as np
import ollama

INDEX_PATH = "data/vector_store/faiss.index"
METADATA_PATH = "data/vector_store/metadata.npy"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = 'llama3'


TOP_K = 5


# Create embedding vector for user query with Ollama
def embed_query(text):
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return np.array(response["embedding"], dtype="float32")

# Load FAISS index and metadata to match the "most similar" vector to a chunk of text
def load_vector_store():
    index = faiss.read_index(INDEX_PATH)
    metadata = np.load(METADATA_PATH, allow_pickle=True).tolist()
    return index, metadata


# Find the top-k "most similar" chunks to the user query, based on vector similarity
def retrieve_chunks(query, index, metadata, k=TOP_K):

    # Generate a vector from the user query, reshape it to be 2 dimensional vector (FAISS expects that)
    query_vector = embed_query(query).reshape(1, -1)

    # In the vector database, search for the top_k most similar vectors to the 'vectorised' user query
    distances, indices = index.search(query_vector, k)

    results = []
    # FAISS gives back a result also optimal for multiple queries, but we only have 1, hence indices[0]
    for i in indices[0]:
        results.append(metadata[i])

    # Return the result, that contains metadata about the most relevant chunks (text, chunk id, source)
    return results


# Build prompt
def build_prompt(context_chunks, question):
    context_text = "\n\n".join(chunk["text"] for chunk in context_chunks)
    prompt = f"""
You are a helpful SAP customer support assistant.
Use ONLY the information in the context below to answer the question.
If the answer is not in the context, say you don't know."

Context:
{context_text}

Question:
{question}

Answer:
"""
    return prompt

# Pass prompt to LLM
def ask_llm(prompt):
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def main():
    print("Loading")
    index, metadata = load_vector_store()

    print("Chatbot ready, type a question (or 'exit').\n")

    while True:
        question = input("Your prompt: ")
        if question.lower() == "exit":
            break

        retrieved_chunks = retrieve_chunks(question, index, metadata)
        prompt = build_prompt(retrieved_chunks, question)
        answer = ask_llm(prompt)

        print("\nChatbot answer:\n", answer, "\n")


if __name__ == "__main__":
    main()