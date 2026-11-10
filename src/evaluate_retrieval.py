
from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "retrieval_test_set.csv"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"

# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_passages"
TOP_K = 5


def main():
    # Load test questions
    test_data = pd.read_csv(TEST_DATA_PATH)

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Connect to ChromaDB
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)

    results = []

    print("\nStarting retrieval evaluation...\n")

    for _, row in test_data.iterrows():
        question = row["question"]
        expected_id = int(row["expected_passage_id"])

        # Generate question embedding
        query_embedding = model.encode(
            question,
            normalize_embeddings=True
        )

        # Retrieve top 5 chunks
        retrieved = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=TOP_K
        )

        retrieved_ids = retrieved["metadatas"][0]
        retrieved_passage_ids = [
            int(metadata["passage_id"])
            for metadata in retrieved_ids
        ]

        # Find the rank of the expected passage
        rank = next(
            (
                position
                for position, passage_id in enumerate(
                    retrieved_passage_ids, start=1
                )
                if passage_id == expected_id
            ),
            None
        )

        results.append({
            "question": question,
            "expected_passage_id": expected_id,
            "retrieved_rank": rank,
            "hit_at_1": int(rank is not None and rank <= 1),
            "hit_at_3": int(rank is not None and rank <= 3),
            "hit_at_5": int(rank is not None and rank <= 5),
            "reciprocal_rank": 1 / rank if rank is not None else 0
        })

        print(f"Question: {question}")
        print(f"Expected passage: {expected_id}")
        print(f"Retrieved rank: {rank if rank else 'Not found in top 5'}")
        print("-" * 60)

    # Calculate evaluation metrics
    results_df = pd.DataFrame(results)

    print("\n========== RETRIEVAL EVALUATION ==========")
    print(f"Total questions: {len(results_df)}")
    print(f"Hit@1: {results_df['hit_at_1'].mean():.2%}")
    print(f"Hit@3: {results_df['hit_at_3'].mean():.2%}")
    print(f"Hit@5: {results_df['hit_at_5'].mean():.2%}")
    print(f"MRR: {results_df['reciprocal_rank'].mean():.4f}")

    # Save detailed results
    output_path = PROJECT_ROOT / "reports" / "retrieval_evaluation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
