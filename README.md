# RAG Learning

A hands-on repository for learning and implementing Retrieval-Augmented Generation (RAG) concepts from the ground up. This repository covers the complete RAG pipeline—from document ingestion and vector databases to advanced retrieval techniques and modern RAG architectures.

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
│   ├── chromadb.ipynb
│   └── data/
│
├── FAISS
│   ├── faiss.ipynb
│   └── faiss_index/
│
├── MultiModal-RAG
│
├── Query-Enhancement
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
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Topics Covered

### Data Processing
- PDF Parsing
- DOCX Parsing
- CSV & Excel Parsing
- JSON Parsing

### Embeddings & Vector Databases
- Ollama Embeddings
- ChromaDB
- FAISS

### Chunking
- Recursive Chunking
- Semantic Chunking

### Retrieval Strategies
- Semantic Search
- Hybrid Search
- Maximum Marginal Relevance (MMR)
- Re-Ranking

### Query Optimization
- Query Enhancement Techniques

### Advanced RAG Architectures
- Agentic RAG
- Corrective RAG (CRAG)
- Multimodal RAG

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
- LangGraph
- Ollama
- ChromaDB
- FAISS
- Nomic Embed Text
- BM25 Retriever
- Tavily Search API

---

## Installation

Clone the repository

```bash
git clone https://github.com/priyanshuu-x/RAG-Learning.git
```

Navigate into the project

```bash
cd RAG-Learning
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Examples

Run any Python file directly.

### Semantic Chunking

```bash
python Semantic-Chunking.py
```

### Agentic RAG

```bash
python Agentic-RAG.py
```

### Corrective RAG

```bash
python Corrective-RAG.py
```

or open the notebooks directly in Jupyter Notebook or VS Code.

---

## License

This repository is intended for educational purposes and hands-on learning of Retrieval-Augmented Generation (RAG).