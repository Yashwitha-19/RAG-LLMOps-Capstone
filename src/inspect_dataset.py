
import pandas as pd

# Dataset file paths
passages_path = "data/raw/passages.parquet"
questions_path = "data/raw/questions.parquet"

# Load datasets
passages = pd.read_parquet(passages_path)
questions = pd.read_parquet(questions_path)

# Function to inspect datasets
def inspect_dataset(df, name):

    print("\n" + "=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)

    # Dataset dimensions
    print("\n1. Dataset Shape:")
    print(df.shape)

    # Column names
    print("\n2. Column Names:")
    print(df.columns.tolist())

    # Data types
    print("\n3. Data Types:")
    print(df.dtypes)

    # Missing values
    print("\n4. Missing Values:")
    print(df.isnull().sum())

    # Duplicate records
    print("\n5. Duplicate Rows:")
    print(df.duplicated().sum())

    # First five records
    print("\n6. First Five Records:")
    print(df.head())

    # Basic information
    print("\n7. Dataset Information:")
    df.info()


# Inspect both datasets
inspect_dataset(passages, "PASSAGES")
inspect_dataset(questions, "QUESTIONS")
