from ingest import load_pdf, chunk_text
from embeddings import create_embeddings
import embeddings
print(dir(embeddings))

text = load_pdf("data/sample.pdf")
chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

print("Total chunks:", len(chunks))
print("Embedding shape:", len(embeddings), len(embeddings[0]))

from retriever import retrieve

query = "What is Garvit's work experience?"

results = retrieve(query, chunks, embeddings)

print("\nTop Results:\n")
for r in results:
    print(r)
    print("-" * 50)