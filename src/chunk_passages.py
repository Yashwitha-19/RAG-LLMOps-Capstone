
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

# File paths
input_path = Path("data/processed/cleaned_passages.parquet")
output_path = Path("data/processed/chunked_passages.parquet")

# Chunking settings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load cleaned passages
df = pd.read_parquet(input_path)

chunks = []

for _, row in df.iterrows():
    passage_id = row["id"]
    text = str(row["passage"])

    # Convert text into tokens
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    start = 0
    chunk_number = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]

        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )

        chunks.append({
            "chunk_id": f"{passage_id}_{chunk_number}",
            "passage_id": passage_id,
            "chunk_number": chunk_number,
            "chunk_text": chunk_text
        })

        chunk_number += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

# Create output DataFrame
chunk_df = pd.DataFrame(chunks)

# Save chunks
output_path.parent.mkdir(parents=True, exist_ok=True)
chunk_df.to_parquet(output_path, index=False)

# Display results
print("Chunking completed!")
print("Original passages:", len(df))
print("Generated chunks:", len(chunk_df))
print("Chunk size:", CHUNK_SIZE)
print("Chunk overlap:", CHUNK_OVERLAP)
print("Saved to:", output_path)

print("\nSample chunks:")
print(chunk_df.head(3).to_string(index=False))
