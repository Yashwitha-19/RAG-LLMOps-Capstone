
import pandas as pd
import numpy as np
import chromadb
from pathlib import Path

# File paths
chunks_path = Path("data/processed/chunked_passages.parquet")
embeddings_path = Path("data/processed/embeddings.npy")
database_path = "data/chroma_db"

# Load chunks and embeddings
df = pd.read_parquet(chunks_path)
embeddings = np.load(embeddings_path)

# Safety check
assert len(df) == len(embeddings), "Chunks and embeddings count do not match!"

# Create persistent Chroma database
client = chromadb.PersistentClient(path=database_path)

# Create or access collection
collection = client.get_or_create_collection(
    name="rag_passages",
    metadata={"hnsw:space": "cosine"}
)

# Insert data in batches
batch_size = 500

for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))

    batch_df = df.iloc[start:end]

    collection.upsert(
        ids=batch_df["chunk_id"].astype(str).tolist(),
        embeddings=embeddings[start:end].tolist(),
        documents=batch_df["chunk_text"].tolist(),
        metadatas=[
            {
                "passage_id": int(row["passage_id"]),
                "chunk_number": int(row["chunk_number"])
            }
            for _, row in batch_df.iterrows()
        ]
    )

    print(f"Inserted {end} / {len(df)} chunks")

print("\nChromaDB creation completed!")
print("Collection name: rag_passages")
print("Total stored chunks:", collection.count())
print("Database location:", database_path)
