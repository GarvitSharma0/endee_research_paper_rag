import os
from sentence_transformers import SentenceTransformer
import endee

# -------------------------------
# INIT MODEL + ENDEE
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Endee DB
db = endee.Client()
collection = db.get_or_create_collection("research_papers")

# -------------------------------
# LOAD TEXT (dummy for now)
# -------------------------------
def load_text():
    with open("sample.txt", "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------
# CHUNKING
# -------------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# -------------------------------
# STORE IN ENDEE
# -------------------------------
def store_embeddings(chunks):
    embeddings = model.encode(chunks)

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[str(i)],
            embeddings=[emb.tolist()],
            documents=[chunk]
        )

# -------------------------------
# SEARCH
# -------------------------------
def search(query, k=3):
    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=k
    )

    return results["documents"][0]

# -------------------------------
# MAIN LOOP
# -------------------------------
if __name__ == "__main__":
    print("Loading document...")
    text = load_text()

    chunks = chunk_text(text)
    store_embeddings(chunks)

    print("Ready! Ask questions.\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        docs = search(query)
        context = "\n".join(docs)

        print("\nRetrieved Context:\n", context)