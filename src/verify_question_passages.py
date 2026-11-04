
import pandas as pd

# Load datasets
passages = pd.read_parquet("data/raw/passages.parquet")
questions = pd.read_parquet("data/raw/questions.parquet")

print("=" * 65)
print("QUESTION AND PASSAGE RELATIONSHIP INVESTIGATION")
print("=" * 65)

# Search the entire passage collection for important keywords
keywords = [
    "Abraham Lincoln",
    "Gettysburg",
    "National Banking Act",
    "John C. Breckinridge"
]

print("\n1. SEARCHING PASSAGES FOR KEYWORDS")

for keyword in keywords:

    matches = passages[
        passages["passage"].str.contains(
            keyword,
            case=False,
            na=False,
            regex=False
        )
    ]

    print(f"\nKeyword: {keyword}")
    print("Matching passages:", len(matches))

    for _, row in matches.head(3).iterrows():
        print("\nPassage ID:", row["id"])
        print("Text:", row["passage"][:400])


# Check whether question IDs map to matching passage content
print("\n2. CHECKING FIRST 10 QUESTIONS")

for _, question_row in questions.head(10).iterrows():

    qid = question_row["id"]

    matched = passages[passages["id"] == qid]

    print("\nQuestion:", question_row["question"])
    print("Expected answer:", question_row["answer"])

    if not matched.empty:
        print("Same-ID passage:")
        print(matched.iloc[0]["passage"][:300])


# Inspect how many passages mention Lincoln
print("\n3. LINCOLN CONTENT CHECK")

lincoln_matches = passages[
    passages["passage"].str.contains(
        "Lincoln",
        case=False,
        na=False,
        regex=False
    )
]

print("Passages mentioning Lincoln:", len(lincoln_matches))

print("\nINVESTIGATION COMPLETED!")
