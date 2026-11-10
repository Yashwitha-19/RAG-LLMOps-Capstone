
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# File paths
input_path = Path("data/processed/chunked_passages.parquet")
output_path = Path("data/processed/embeddings.npy")

# Load chunked dataset
df = pd.read_parquet(input_path)

print("Total chunks:", len(df))

# Load embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"

print("\nLoading embedding model...")
model = SentenceTransformer(model_name)

# Generate embeddings
print("\nGenerating embeddings...")
embeddings = model.encode(
    df["chunk_text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Save embeddings
np.save(output_path, embeddings)

print("\nEmbedding generation completed!")
print("Embedding shape:", embeddings.shape)
print("Saved to:", output_path)
