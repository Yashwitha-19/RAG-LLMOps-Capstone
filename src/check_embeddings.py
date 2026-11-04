import numpy as np

embeddings = np.load("data/processed/embeddings.npy")

print("Shape:", embeddings.shape)
print("Data type:", embeddings.dtype)

print("Contains NaN:", np.isnan(embeddings).any())
print("Contains Infinity:", np.isinf(embeddings).any())

print("First embedding:")
print(embeddings[0][:10])

print("Embedding norms:")
print(np.linalg.norm(embeddings[:5], axis=1))