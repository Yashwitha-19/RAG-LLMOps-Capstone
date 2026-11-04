
import pandas as pd
import re
from pathlib import Path

# File paths
input_path = Path("data/raw/passages.parquet")
output_path = Path("data/processed/cleaned_passages.parquet")

# Load original dataset
df = pd.read_parquet(input_path)

print("Original dataset:", df.shape)

# Clean passage text
def clean_text(text):
    text = str(text)

    # Remove extra spaces and line breaks
    text = re.sub(r"\s+", " ", text)

    # Remove spaces at the beginning and end
    text = text.strip()

    return text

df["passage"] = df["passage"].apply(clean_text)

# Remove empty passages
df = df[df["passage"].str.len() > 0]

# Preserve original passage IDs
df["id"] = df["id"].astype(int)

# Save cleaned dataset
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_path, index=False)

print("\nCleaning completed!")
print("Cleaned dataset:", df.shape)
print("Saved to:", output_path)
print("\nSample cleaned passages:")
print(df.head(3).to_string(index=False))
