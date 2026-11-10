# End-to-End LLMOps Framework for a Reliable RAG-Based Question-Answering System

## 1. Project Overview

This project focuses on developing an end-to-end Retrieval-Augmented Generation (RAG) system using Large Language Model Operations (LLMOps) practices.

The system retrieves relevant information from a document collection and uses that information as evidence to answer user questions. The project aims to improve the reliability, reproducibility, evaluation and monitoring of RAG-based question-answering applications.

## 2. Project Objectives

- Build a document preprocessing and chunking pipeline.
- Generate semantic embeddings from text passages.
- Store embeddings in a vector database.
- Retrieve relevant passages based on user questions.
- Evaluate retrieval performance using standard metrics.
- Develop an end-to-end RAG application.
- Introduce LLMOps practices such as experiment tracking, automated testing and monitoring.

## 3. Dataset

**Dataset:** RAG-Mini-Wikipedia

**Source:** [Hugging Face – RAG Mini Wikipedia](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)

The dataset contains Wikipedia passages and question-answer pairs.

Current dataset inspection:

| Dataset | Records |
|---|---:|
| Passages | 3,200 |
| Questions | 918 |

The raw dataset files are stored locally and are excluded from GitHub.

## 4. Technical Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Development environment | VS Code |
| Data processing | Pandas, NumPy |
| Embedding model | all-MiniLM-L6-v2 |
| Embedding library | Sentence Transformers |
| Vector database | ChromaDB |
| Dataset format | Parquet |
| Version control | Git and GitHub |

Additional technologies for the upcoming development stages may include Streamlit, MLflow, Pytest and GitHub Actions.

## 5. System Workflow

1. Load and inspect the Wikipedia dataset.
2. Clean and preprocess passage text.
3. Divide passages into smaller text chunks.
4. Generate vector embeddings.
5. Store the embeddings in ChromaDB.
6. Convert user questions into embeddings.
7. Retrieve the most relevant passages.
8. Use retrieved evidence to support generated answers.
9. Evaluate retrieval and answer quality.

## 6. Project Structure

```text
RAG-LLMOps-Capstone/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma_db/
│
├── reports/
│   └── retrieval_evaluation.csv
│
├── src/
│   ├── check_chunks.py
│   ├── check_embeddings.py
│   ├── chunk_passages.py
│   ├── clean_passages.py
│   ├── data_quality_analysis.py
│   ├── embeddings.py
│   ├── evaluate_retrieval.py
│   ├── evidence_mapping.py
│   ├── inspect_dataset.py
│   ├── retrieval.py
│   ├── vector_database.py
│   └── verify_question_passages.py
│
├── tests/
├── app/
├── notebooks/
├── requirements.txt
└── README.md
```

Note: The data directory and local environment are excluded from version control.

## 7. Installation and Setup

Clone the repository:

```bash
git clone https://github.com/Samanvai22/RAG-LLMOps-Capstone.git
cd RAG-LLMOps-Capstone
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from Hugging Face and place the required Parquet files inside `data/raw/`:

- passages.parquet
- questions.parquet

Run the scripts from the project root in the appropriate pipeline order.

## 8. Preliminary Retrieval Evaluation

The initial retrieval evaluation uses five manually verified questions.

| Metric | Result |
|---|---:|
| Hit@1 | 60% |
| Hit@3 | 100% |
| Hit@5 | 100% |
| Mean Reciprocal Rank (MRR) | 0.7667 |

These results are preliminary and based on a small test set. A larger evaluation dataset is required for a more reliable performance assessment.

## 9. Current Development Status

**Completed:**
- Dataset inspection and quality analysis.
- Passage cleaning.
- Text chunking.
- Embedding generation.
- ChromaDB vector storage.
- Semantic retrieval.
- Initial retrieval evaluation.

**Planned:**
- Expand the verified evaluation dataset.
- Integrate a local or accessible language model.
- Develop a RAG-based question-answering interface.
- Evaluate answer quality and citation reliability.
- Implement experiment tracking.
- Add automated testing and CI/CD.
- Introduce performance monitoring.

## 10. Project Goal

The final goal is to develop a reliable and reproducible RAG-based question-answering application supported by an end-to-end LLMOps workflow.

---

**Project:** MSc Data Science Capstone  
**Repository:** [RAG-LLMOps-Capstone](https://github.com/Samanvai22/RAG-LLMOps-Capstone)