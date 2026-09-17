# Wikipedia RAG

A simple Retrieval-Augmented Generation (RAG) application that uses **Wikipedia as the external knowledge source** and an **open-source LLM** to answer user questions.

The first version is intentionally designed as a **CLI application**. The goal is to understand the core RAG pipeline before introducing additional frameworks or a web UI.

## Overview

The application follows this flow:

```text
Wikipedia
    │
    ▼
Fetch Documents
    │
    ▼
Process & Chunk Text
    │
    ▼
Generate Embeddings
    │
    ▼
FAISS Vector Store
    │
    │
    └──────────────┐
                   │
User Question     │
    │              │
    ▼              │
Generate Query     │
Embedding          │
    │              │
    ▼              │
FAISS Retrieval ◄──┘
    │
    ▼
Relevant Context
    │
    ▼
Prompt + Context
    │
    ▼
Open-Source LLM
    │
    ▼
Answer
```

## Goals

The main goals of this project are to understand:

* Retrieval-Augmented Generation (RAG)
* Document loading and processing
* Text chunking
* Embeddings
* Vector databases
* Semantic search
* Prompt construction
* Open-source LLM inference
* How retrieval and generation work together

The initial implementation avoids unnecessary complexity and does not require LangChain.

LangChain can be introduced later to compare its abstractions with the manually implemented pipeline.

---

## Project Structure

```text
wiki-rag/
│
├── app/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── wikipedia.py
│   │   └── processor.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── faiss_store.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── model.py
│   │
│   └── rag/
│       ├── __init__.py
│       └── pipeline.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── vector_db/
│
├── tests/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Module Responsibilities

### `app/cli.py`

Provides the command-line interface using **Click**.

Example commands:

```bash
wiki-rag fetch "Artificial Intelligence"
wiki-rag index
wiki-rag ask "What is artificial intelligence?"
```

The CLI should only handle user interaction and invoke the appropriate application components.

---

### `app/data/wikipedia.py`

Responsible for fetching data from Wikipedia.

Responsibilities:

* Query Wikipedia
* Retrieve article content
* Extract title and text
* Store raw documents

Example:

```text
Wikipedia
    ↓
Article
    ↓
Raw document
```

---

### `app/data/processor.py`

Responsible for preparing documents for embedding.

Responsibilities:

* Clean text
* Remove unnecessary content
* Split documents into smaller chunks
* Maintain document metadata

Example:

```text
Wikipedia Article
       ↓
Clean Text
       ↓
Chunks
```

---

### `app/embeddings/embedder.py`

Responsible for converting text into numerical vectors.

Example:

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

A Sentence Transformers model can be used initially.

---

### `app/vectorstore/faiss_store.py`

Responsible for managing the FAISS vector index.

Responsibilities:

* Add embeddings
* Store document metadata
* Save the index
* Load the index
* Perform similarity searches

Example:

```text
Query Vector
     ↓
   FAISS
     ↓
Top-K Relevant Chunks
```

---

### `app/llm/model.py`

Responsible for loading and interacting with the open-source LLM.

Responsibilities:

* Load the model
* Load the tokenizer
* Generate responses
* Hide model-specific implementation details from the rest of the application

The rest of the application should interact with the LLM through a simple interface such as:

```python
llm.generate(prompt)
```

---

### `app/rag/pipeline.py`

This is the main RAG orchestration layer.

It connects:

* Query embedding
* Vector retrieval
* Context construction
* Prompt construction
* LLM generation

The basic flow is:

```text
User Question
      ↓
Query Embedding
      ↓
FAISS Search
      ↓
Relevant Documents
      ↓
Build Prompt
      ↓
LLM
      ↓
Answer
```

---

### `app/config.py`

Contains application configuration such as:

* Model names
* Vector database paths
* Data paths
* Retrieval parameters
* Chunk size
* Chunk overlap
* Number of documents to retrieve

Keeping configuration separate makes it easier to change models and parameters later.

---

## Data Flow

### 1. Fetch

```bash
wiki-rag fetch "Artificial Intelligence"
```

```text
Wikipedia
    ↓
wikipedia.py
    ↓
data/raw/
```

### 2. Index

```bash
wiki-rag index
```

```text
Raw Documents
    ↓
processor.py
    ↓
Chunks
    ↓
embedder.py
    ↓
Embeddings
    ↓
faiss_store.py
    ↓
FAISS Index
```

### 3. Ask

```bash
wiki-rag ask "What is artificial intelligence?"
```

```text
Question
    ↓
Embedding
    ↓
FAISS
    ↓
Relevant Chunks
    ↓
Prompt
    ↓
LLM
    ↓
Answer
```

---

## Initial Technology Stack

| Component     | Technology                |
| ------------- | ------------------------- |
| Language      | Python                    |
| CLI           | Click                     |
| Data Source   | Wikipedia                 |
| Embeddings    | Sentence Transformers     |
| Vector Store  | FAISS                     |
| LLM           | Hugging Face Transformers |
| Deep Learning | PyTorch                   |
| HTTP          | Requests                  |

LangChain is **not required for the initial implementation**.

It can be introduced in a later version to provide abstractions for document loading, splitting, embeddings, retrieval, prompts, and chains.

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Example Usage

Fetch a Wikipedia article:

```bash
wiki-rag fetch "Python (programming language)"
```

Create the vector index:

```bash
wiki-rag index
```

Ask a question:

```bash
wiki-rag ask "Who created Python?"
```

The application retrieves relevant information from the indexed Wikipedia content and provides it as context to the LLM.

---

## Future Improvements

The initial version will focus only on the basic RAG pipeline.

Possible future improvements include:

* LangChain integration
* Better document chunking
* Metadata filtering
* Hybrid search
* Reranking
* Conversation history
* Multiple data sources
* Streaming responses
* FastAPI backend
* React/Streamlit UI
* Chat history
* Evaluation of retrieval quality
* RAG evaluation metrics
* Support for larger local LLMs

The UI can be added later without significantly changing the core RAG pipeline:

```text
             ┌──────────────┐
             │     CLI      │
             └──────┬───────┘
                    │
             ┌──────▼───────┐
             │ RAG Pipeline │
             └──────▲───────┘
                    │
             ┌──────┴───────┐
             │     UI       │
             └──────────────┘
```

The core idea is to keep **data ingestion, retrieval, generation, and user interfaces separate**, allowing each part to evolve independently.
