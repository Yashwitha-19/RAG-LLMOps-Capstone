
import pandas as pd

# Load datasets
passages = pd.read_parquet("data/raw/passages.parquet")
questions = pd.read_parquet("data/raw/questions.parquet")

print("=" * 60)
print("DETAILED DATA QUALITY ANALYSIS")
print("=" * 60)

# 1. Text length analysis
passages["passage_length"] = passages["passage"].str.len()
questions["question_length"] = questions["question"].str.len()
questions["answer_length"] = questions["answer"].str.len()

print("\n1. PASSAGE LENGTH STATISTICS")
print(passages["passage_length"].describe())

print("\n2. QUESTION LENGTH STATISTICS")
print(questions["question_length"].describe())

print("\n3. ANSWER LENGTH STATISTICS")
print(questions["answer_length"].describe())

# 2. Empty or whitespace-only text
print("\n4. EMPTY TEXT CHECK")
print("Empty passages:",
      passages["passage"].str.strip().eq("").sum())

print("Empty questions:",
      questions["question"].str.strip().eq("").sum())

print("Empty answers:",
      questions["answer"].str.strip().eq("").sum())

# 3. Duplicate text check
print("\n5. DUPLICATE TEXT CHECK")
print("Duplicate passages:",
      passages["passage"].duplicated().sum())

print("Duplicate questions:",
      questions["question"].duplicated().sum())

# 4. ID analysis
print("\n6. ID ANALYSIS")

print("Passage ID range:",
      passages["id"].min(), "to", passages["id"].max())

print("Question ID range:",
      questions["id"].min(), "to", questions["id"].max())

print("Unique passage IDs:", passages["id"].nunique())
print("Unique question IDs:", questions["id"].nunique())

# Check ID overlap without assuming a relationship
overlap = set(passages["id"]) & set(questions["id"])

print("Number of overlapping IDs:", len(overlap))
print("First 10 overlapping IDs:", sorted(overlap)[:10])

# 5. Display longest passages
print("\n7. THREE LONGEST PASSAGES")
print(
    passages.nlargest(3, "passage_length")[
        ["id", "passage_length"]
    ]
)

# 6. Display shortest passages
print("\n8. THREE SHORTEST PASSAGES")
print(
    passages.nsmallest(3, "passage_length")[
        ["id", "passage_length"]
    ]
)

print("\nDATA QUALITY ANALYSIS COMPLETED!")
