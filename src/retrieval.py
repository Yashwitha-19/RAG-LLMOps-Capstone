
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to existing ChromaDB
client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    name="rag_passages"
)

# Ask a question
question = input("Enter your question: ")

# Convert question into embedding
query_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

# Search ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# Display results
print("\nTop 5 Retrieved Passages:\n")

for i in range(len(results["ids"][0])):
    print(f"Result {i + 1}")
    print("Chunk ID:", results["ids"][0][i])
    print("Passage ID:", results["metadatas"][0][i]["passage_id"])
    print("Distance:", results["distances"][0][i])
    print("Text:", results["documents"][0][i])
    print("-" * 80)
