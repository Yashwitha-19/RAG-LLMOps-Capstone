import pandas as pd

df = pd.read_parquet("data/processed/chunked_passages.parquet")

print("Total chunks:", len(df))
print("\nColumns:", df.columns.tolist())

print("\nChunks per original passage:")
print(df.groupby("passage_id").size().describe())

print("\nFirst 5 chunks:")
print(df.head().to_string(index=False))

print("\nMissing values:")
print(df.isnull().sum())