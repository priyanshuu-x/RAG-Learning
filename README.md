# RAG Learning

A hands-on repository for learning and implementing Retrieval-Augmented Generation (RAG) concepts from the ground up. This repository demonstrates the core building blocks of modern RAG systems, including document ingestion, parsing, embeddings, vector databases, and advanced retrieval strategies.

---

## Repository Structure

```text
RAG
│
├── 0-Data-Ingestion-and-Parsing
│   ├── PDF Parsing
│   ├── DOCX Parsing
│   ├── CSV & Excel Parsing
│   ├── JSON Parsing
│   └── data/
│
├── 1-VectorEmbeddings
│   └── embedding.ipynb
│
├── 2-Vector Stores
│   ├── 1-chromadb.ipynb
│   └── data/
│
├── FAISS
│   ├── faiss.ipynb
│   └── faiss_index/
│
├── Search-Strategies
│   ├── Semantic-Chunking.py
│   ├── Hybrid-Search.py
│   ├── MMR.py
│   ├── ReRanking.py
│   ├── sample_document.txt
│   └── requirements.txt
│
├── main.py
├── requirements.txt
└── pyproject.toml
```

---

## Features

- Document ingestion from multiple file formats
- PDF, DOCX, CSV, Excel, and JSON parsing
- Recursive and semantic text chunking
- Embedding generation using Ollama
- ChromaDB vector database integration
- FAISS vector search
- Semantic search
- Hybrid Search (Dense + BM25 Retrieval)
- Maximum Marginal Relevance (MMR) Retrieval
- Re-Ranking for improved retrieval quality

---

## Technologies Used

- Python
- LangChain
- LangChain Community
- LangChain Chroma
- LangChain Ollama
- ChromaDB
- FAISS
- Ollama
- Nomic Embed Text
- BM25 Retriever

---

## Installation

Clone the repository:

```bash
git clone https://github.com/priyanshuu-x/RAG-Learning.git
```

Navigate to the project directory:

```bash
cd RAG-Learning
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Examples

Run any notebook or Python script from its respective directory.

Example:

```bash
python Search-Strategies/Hybrid-Search.py
```

or open the notebooks directly in Jupyter Notebook or VS Code.

---

## License

This repository is intended for educational purposes and hands-on learning of Retrieval-Augmented Generation (RAG).
