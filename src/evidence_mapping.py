
import pandas as pd

# Load datasets
passages = pd.read_parquet("data/raw/passages.parquet")
questions = pd.read_parquet("data/raw/questions.parquet")

# Search terms for the first 10 questions
search_terms = {
    0: ["Abraham Lincoln", "sixteenth President"],
    2: ["National Banking Act", "Lincoln"],
    4: ["Lincoln mother", "Nancy Hanks"],
    6: ["Lincoln education", "eighteen months"],
    8: ["Lincoln political career", "1832"],
    10: ["Legal Tender Act", "United States Note"],
    12: ["Grace Bedell", "Lincoln beard"],
    14: ["Gettysburg Address", "1776"],
    16: ["John C. Breckinridge", "1860 election"],
    18: ["first President", "George Washington"]
}

results = []

for _, qrow in questions.head(10).iterrows():

    qid = qrow["id"]

    print("\n" + "=" * 60)
    print("QUESTION ID:", qid)
    print("QUESTION:", qrow["question"])
    print("EXPECTED ANSWER:", qrow["answer"])

    terms = search_terms.get(qid, [])

    for term in terms:

        matches = passages[
            passages["passage"].str.contains(
                term,
                case=False,
                na=False,
                regex=False
            )
        ]

        for _, prow in matches.iterrows():

            results.append({
                "question_id": qid,
                "question": qrow["question"],
                "expected_answer": qrow["answer"],
                "search_term": term,
                "candidate_passage_id": prow["id"],
                "candidate_passage": prow["passage"],
                "verification_status": "Pending"
            })

print("\nTotal candidate matches:", len(results))

# Export candidates for manual verification
output = pd.DataFrame(results)

output.to_csv(
    "data/processed/evidence_candidates.csv",
    index=False
)

print("\nSaved evidence candidates to:")
print("data/processed/evidence_candidates.csv")
