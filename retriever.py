import numpy as np
from embeddings import create_embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, chunks, embeddings, top_k=3):
    # Convert query to embedding
    query_embedding = create_embeddings([query])[0]

    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, chunks[i]))

    # Sort by similarity
    scores.sort(reverse=True, key=lambda x: x[0])

    return [chunk for _, chunk in scores[:top_k]]