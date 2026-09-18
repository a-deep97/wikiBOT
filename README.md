# Wikipedia RAG

A simple Retrieval-Augmented Generation (RAG) application that uses **Wikipedia as the external knowledge source** and an **open-source LLM** to answer user questions.

The first version is intentionally designed as a **CLI application**. The goal is to understand the core RAG pipeline before introducing additional frameworks or a web UI.

---

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
                    SQLite Knowledge Store
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
User Question                             │
      │                                    │
      ▼                                    │
Generate Query                             │
Embedding                                  │
      │                                    │
      ▼                                    │
FAISS Retrieval ◄──────────────────────────┘
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

SQLite and FAISS have different responsibilities:

```text
SQLite
"What knowledge do we already have?"

FAISS
"Which knowledge is relevant to this question?"
```

SQLite acts as the persistent local knowledge store, while FAISS is used for semantic retrieval.

---

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
* Persistent knowledge storage
* Caching retrieved documents
* How retrieval and generation work together

The initial implementation avoids unnecessary complexity and does not require LangChain.

LangChain can be introduced later to compare its abstractions with the manually implemented pipeline.

---

## Project Structure

```text
wikiBOT/
│
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── pipeline.py
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
│   ├── vector/
│   │   ├── __init__.py
│   │   └── faiss_store.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── sqlite_store.py
│   │
│   └── knowledge/
│       ├── __init__.py
│       └── manager.py
│
├── data/
│   └── wikibot.db
│
├── tests/
│   └── test_database.py
│
├── requirements.txt
└── README.md
```

---

## Module Responsibilities

### `src/cli.py`

Provides the command-line interface using **Click**.

The CLI is responsible for:

* Getting the Wikipedia topic from the user
* Accepting questions
* Starting a new topic
* Exiting the application
* Displaying answers

The CLI should only handle user interaction and invoke the appropriate application components.

---

### `src/data/wikipedia.py`

Responsible for fetching data from Wikipedia.

Responsibilities:

* Query Wikipedia
* Retrieve article content
* Extract article title
* Extract introduction
* Extract sections
* Extract links to other Wikipedia articles

Example:

```text
Wikipedia
    ↓
Article
    ├── Title
    ├── Introduction
    ├── Sections
    └── Links
```

Unnecessary sections such as references and external links can be excluded during processing.

---

### `src/data/processor.py`

Responsible for preparing documents for embedding.

Responsibilities:

* Clean text
* Remove unnecessary whitespace
* Split documents into smaller chunks
* Maintain document metadata

Each chunk contains metadata such as:

```text
Article
Section
Chunk Index
Content
```

Example:

```text
Article: Python

Section: History

Python was created by Guido van Rossum...
```

This metadata will later allow the application to identify the source of retrieved information.

---

### `src/embeddings/embedder.py`

Responsible for converting text into numerical vectors.

Example:

```text
Text
  ↓
Embedding Model
  ↓
Vector
```

The current implementation uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model generates 384-dimensional embeddings.

---

### `src/vector/faiss_store.py`

Responsible for managing the FAISS vector index.

Responsibilities:

* Add embeddings
* Store document/chunk metadata
* Perform similarity searches
* Return the most relevant chunks

The current implementation uses normalized vectors with inner-product similarity, which is equivalent to cosine similarity.

Example:

```text
Query Vector
     ↓
   FAISS
     ↓
Top-K Relevant Chunks
```

FAISS is responsible for **semantic retrieval**, not persistent knowledge storage.

---

### `src/models/model.py`

Responsible for loading and interacting with the open-source LLM.

Responsibilities:

* Load the model
* Load the tokenizer
* Generate responses
* Hide model-specific implementation details from the rest of the application

The rest of the application interacts with the model through a simple interface such as:

```python
model.generate(prompt)
```

The current implementation uses a Hugging Face Transformers model.

---

### `src/database/sqlite_store.py`

Responsible for persistent local storage.

SQLite stores the processed Wikipedia knowledge so that articles do not need to be downloaded and processed every time they are requested.

The database is stored at:

```text
data/wikibot.db
```

The database contains three main tables:

#### `articles`

Stores article information:

```text
id
title
fetched_at
```

#### `chunks`

Stores processed article chunks:

```text
id
article_id
section
chunk_index
content
```

#### `links`

Stores links between Wikipedia articles:

```text
id
source_article_id
target_title
```

The relationship is:

```text
Article
   │
   ├── Chunks
   │
   └── Links → Other Wikipedia Articles
```

---

### `src/knowledge/manager.py`

The Knowledge Manager controls how wikiBOT obtains knowledge.

Its main responsibility is to decide whether to use the local SQLite cache or fetch information from Wikipedia.

The basic flow is:

```text
Request Article
      │
      ▼
Check SQLite
   ┌──┴──┐
   │     │
Found  Not Found
   │     │
   ▼     ▼
Cache  Wikipedia
          │
          ▼
       Process
          │
          ▼
        SQLite
```

This prevents repeated downloading and processing of the same article.

---

## Data Flow

### 1. Fetch

When an article is requested:

```text
Wikipedia
    ↓
wikipedia.py
    ↓
Article
```

The article is then processed into structured chunks.

---

### 2. Cache

Before fetching an article, wikiBOT checks SQLite.

```text
Requested Article
       ↓
   SQLite Check
      /     \
    Yes      No
     ↓        ↓
  Load DB   Wikipedia
              ↓
           Process
              ↓
           Save DB
```

This makes SQLite the local persistent knowledge cache.

---

### 3. Process

```text
Wikipedia Article
       ↓
Clean Text
       ↓
Sections
       ↓
Chunks + Metadata
```

Example chunk:

```python
{
    "section": "History",
    "chunk_index": 2,
    "content": "Article: Python\nSection: History\n..."
}
```

---

### 4. Index

The chunk content is converted into embeddings:

```text
Chunks
   ↓
embedder.py
   ↓
Embeddings
   ↓
faiss_store.py
   ↓
FAISS
```

The chunk metadata is retained so that the retrieved result still contains information about its source.

---

### 5. Ask

```text
Question
    ↓
Query Embedding
    ↓
FAISS
    ↓
Relevant Chunks
    ↓
Build Prompt
    ↓
LLM
    ↓
Answer
```

---

## SQLite Cache

The SQLite cache is designed to avoid unnecessary calls to Wikipedia.

For example, the first time the user loads:

```text
Python
```

wikiBOT performs:

```text
Python
  ↓
Wikipedia
  ↓
Process
  ↓
SQLite
```

The next time the same article is requested:

```text
Python
  ↓
SQLite
  ↓
Cached Chunks
```

No new Wikipedia download is required.

The stored article also contains its Wikipedia links. These links will later be used to expand the local knowledge base by checking which linked articles are already cached and which ones still need to be fetched.

---

## Initial Technology Stack

| Component             | Technology                |
| --------------------- | ------------------------- |
| Language              | Python                    |
| CLI                   | Click                     |
| Data Source           | Wikipedia                 |
| Local Knowledge Store | SQLite                    |
| Embeddings            | Sentence Transformers     |
| Vector Store          | FAISS                     |
| LLM                   | Hugging Face Transformers |
| Deep Learning         | PyTorch                   |
| HTTP                  | Requests                  |

SQLite is built into Python and therefore does not require an additional dependency.

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

Run the application:

```bash
python -m src.cli
```

The CLI asks for a Wikipedia page:

```text
Enter Wikipedia Page Title: Python
```

After loading the article, questions can be asked:

```text
Your question: Who created Python?
```

To load a different topic:

```text
Your question: new
```

To exit:

```text
Your question: exit
```

The application retrieves relevant information from the indexed Wikipedia content and provides it as context to the LLM.

---

## Current Development Status

Implemented:

* Wikipedia article fetching
* Wikipedia section extraction
* Wikipedia link extraction
* Text cleaning
* Section-aware chunking
* Chunk metadata
* Sentence Transformer embeddings
* FAISS semantic search
* Cosine-similarity retrieval
* Hugging Face LLM integration
* RAG pipeline
* SQLite database
* Local article caching
* Persistent article chunks
* Persistent Wikipedia links
* Knowledge Manager

Current architecture:

```text
                    ┌───────────────┐
                    │   Wikipedia   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Data Processor│
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │    SQLite     │
                    │   Knowledge   │
                    │     Store     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Embeddings  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     FAISS     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │      LLM      │
                    └───────┬───────┘
                            │
                            ▼
                         Answer
```

---

## Future Improvements

The initial version focuses on understanding the basic RAG pipeline.

Possible future improvements include:

* Linked-article cache expansion
* Better linked-page selection
* Persistent FAISS index
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
* LangChain integration

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
             │      UI      │
             └──────────────┘
```

The core idea is to keep **data ingestion, knowledge storage, retrieval, generation, and user interfaces separate**, allowing each part to evolve independently.
