# wikiBOT

A Python-based Retrieval-Augmented Generation (RAG) application that lets users ask questions about Wikipedia articles and receive context-aware answers from a selected language model. wikiBOT combines semantic search, keyword-based retrieval, persistent knowledge storage, and a Streamlit interface.

## Overview

wikiBOT retrieves relevant information from Wikipedia articles and supplies that context to a language model to generate answers.

Instead of relying solely on the model's internal knowledge, the application retrieves information from the selected article before generating a response.

### Key Features

- **Wikipedia integration:** Fetch articles, summaries, sections, and links.
- **Text processing:** Clean article content and split it into sentence-aware chunks.
- **Semantic retrieval:** Generate embeddings using Sentence Transformers and search with FAISS.
- **Keyword retrieval:** Use BM25 to retrieve relevant passages based on keyword matches.
- **Hybrid retrieval:** Combine semantic and keyword-based retrieval candidates.
- **Similarity filtering:** Filter semantic search results using a configurable similarity threshold.
- **Persistent storage:** Cache article content, chunks, links, and sections in SQLite.
- **Knowledge management:** Reuse cached article data and expand knowledge through Wikipedia links.
- **Multiple language models:** Select from supported causal and sequence-to-sequence models.
- **GPU support:** Support CUDA-enabled model execution when compatible hardware and dependencies are available.
- **Shared backend:** Centralize model resources, storage, retrieval, and RAG orchestration.
- **Streamlit interface:** Search for Wikipedia articles using autocomplete, automatically load selected articles, read article sections, and interact with the chatbot.
- **Source-backed answers:** Every chat answer includes a collapsible "Sources" panel showing the article sections and text snippets that were used as context.
- **Chat conveniences:** Clickable starter questions, a status bar showing the article, model and device, and Clear and Export (Markdown) controls.
- **QnA interface:** Provide a dedicated UI section for saved question-and-answer pairs; persistent QnA storage is planned.

## Architecture

```text
                         Wikipedia
                             |
                             v
                   Article Retrieval
                             |
                             v
                  Text Cleaning and Parsing
                             |
                             v
                    Sentence-aware Chunking
                             |
                             v
                       SQLite Storage
                  Articles / Chunks / Links
                         /         \
                        v           v
                  Embeddings    BM25 Index
                        |           |
                        v           |
                      FAISS         |
                        \           /
                         \         /
                          v       v
                       Hybrid Retrieval
                             |
                             v
                    Candidate Selection
                             |
                             v
                      RAG Pipeline
                             |
                  Retrieved Context + Query
                             |
                             v
                      Language Model
                             |
                             v
                       Final Answer
```

The application uses a shared `WikiBotBackend` to coordinate the main components. The CLI and Streamlit interface interact with this backend instead of independently managing model resources, databases, and retrieval components.

## Project Structure

```text
wikiBOT/
├── src/
│   ├── cli.py
│   ├── config.py
│   ├── pipeline.py
│   ├── backend/
│   │   ├── wiki_backend.py
│   │   └── article_context.py
│   ├── app_web/
│   │   ├── wikibot.py
│   │   ├── helpers.py
│   │   ├── styles.py
│   │   ├── search_history.py
│   │   └── components/
│   │       ├── sidebar.py
│   │       ├── search_panel.py
│   │       ├── article_reader.py
│   │       ├── chat_interface.py
│   │       └── qna_panel.py
│   ├── data/
│   │   ├── wikipedia.py
│   │   └── processor.py
│   ├── embeddings/
│   │   └── embedder.py
│   ├── vector/
│   │   ├── faiss_store.py
│   │   └── bm25_store.py
│   ├── models/
│   │   ├── data.py
│   │   ├── model.py
│   │   └── shared_resources.py
│   ├── database/
│   │   └── sqlite_store.py
│   └── knowledge/
│       └── manager.py
├── tests/
│   └── test_database.py
├── data/
│   └── wikibot.db        (created at runtime)
└── README.md
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/cli.py` | Command-line interaction with wikiBOT |
| `src/pipeline.py` | RAG pipeline orchestration; `ask()` returns an answer and `ask_with_sources()` also returns the retrieved chunks |
| `src/backend/wiki_backend.py` | Shared backend: resource lifecycle, article loading, autocomplete, and question answering |
| `src/backend/article_context.py` | Retrieval stores and pipeline for one loaded article |
| `src/app_web/wikibot.py` | Streamlit entry point, model selection, and article loading |
| `src/app_web/components/` | Streamlit UI sections: sidebar, search, article reader, chat, and Q&A |
| `src/app_web/helpers.py` | Reusable UI helpers: article rendering, chat source cleanup, status chips, and Q&A UI wrappers |
| `src/app_web/styles.py` | Shared CSS and page header |
| `src/app_web/search_history.py` | Recent article search history |
| `src/data/wikipedia.py` | Fetch Wikipedia articles, summaries, sections, and links |
| `src/data/processor.py` | Clean, process, and chunk article content |
| `src/embeddings/embedder.py` | Generate text embeddings |
| `src/vector/faiss_store.py` | Semantic vector search using FAISS |
| `src/vector/bm25_store.py` | Keyword-based retrieval using BM25 |
| `src/models/data.py` | Model configuration and available model definitions |
| `src/models/model.py` | Language model loading and inference |
| `src/models/shared_resources.py` | Cached embedder and language model shared across the app |
| `src/database/sqlite_store.py` | Persistent SQLite storage and retrieval |
| `src/knowledge/manager.py` | Coordinate article loading, caching, and knowledge preparation |

## Technology Stack

- **Language:** Python
- **Web interface:** Streamlit
- **Wikipedia data:** Wikipedia API integration
- **Embeddings:** Sentence Transformers
- **Vector search:** FAISS
- **Keyword search:** BM25
- **Database:** SQLite
- **Language models:** Supported Hugging Face causal and sequence-to-sequence models
- **Hardware acceleration:** PyTorch and CUDA, when available

## Installation

Clone or download the repository and navigate to the project root.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the project's dependencies:

```bash
python -m pip install -r requirements.txt
```

The exact dependency versions should be maintained in the project's requirements file. GPU users should ensure their PyTorch and CUDA configuration is compatible with their hardware.

## Running wikiBOT

### Streamlit application

From the project root, run:

```bash
streamlit run src/app_web/wikibot.py
```

The application provides a Wikipedia article search interface, an article reader, a chatbot, and a QnA section.

### CLI

The command-line entry point is:

```bash
python -m src.cli
```

Use the CLI options supported by the current implementation to select a model and interact with the application.

## Application Workflow

### 1. Select a Wikipedia article

Use the search box in the Streamlit sidebar to search for an article. Suggestions are retrieved through Wikipedia search and cached temporarily in memory to reduce repeated requests.

Selecting an article automatically triggers knowledge loading.

### 2. Load or reuse article knowledge

The knowledge manager checks SQLite for cached article data. If the article is already cached, its chunks can be reused instead of downloading and processing the article again.

Article sections are stored separately so that the original article can be displayed in the interface while chunks remain available for retrieval.

### 3. Retrieve relevant context

When a question is submitted, wikiBOT searches for relevant chunks using semantic and keyword retrieval.

- FAISS finds semantically similar chunks using embeddings.
- BM25 finds chunks with relevant keyword matches.
- Candidate selection and similarity filtering help determine which chunks are passed to the RAG pipeline.

### 4. Generate an answer

The RAG pipeline combines the user's question with the retrieved context and sends the resulting prompt to the selected language model.

The model generates an answer based on the available context. The chat calls `ask_with_sources()`, so each answer comes back together with the chunks that were used to build its prompt.

### Chat features

- **Sources:** Each answer has a collapsible "Sources" panel listing the article section and a short snippet for every retrieved chunk. The "Article:" and "Section:" header lines added during chunking are stripped for display.
- **Starter questions:** When a conversation is empty, the chat suggests a summary question plus questions based on the article's longest sections. Clicking one submits it.
- **Status bar:** Small chips show the loaded article, the selected model, and whether it runs on GPU or CPU.
- **Clear and Export:** Clear starts a fresh conversation; Export downloads the conversation as Markdown, including the source sections for each answer.
- **Add to Q&A:** Each answer has an "Add to Q&A" button. The click is registered and the button switches to a "Saved" state, but it still calls the placeholder `add_qna` helper (see Save Q&A records below), so nothing is persisted yet.

### 5. Read the source article

The sidebar displays the loaded article's sections in collapsible panels. Article sections are stored separately from retrieval chunks to support both reading and question answering.

### 6. Save Q&A records

The interface includes a QnA section and a UI wrapper for submitting records containing the article title, question, and answer.

The actual backend operation and persistent QnA storage are not yet implemented: `add_qna` and `get_qna` in `src/app_web/helpers.py` are placeholders that only print to the terminal. Once the backend is added, the QnA section will retrieve saved records through it and display them.

## Retrieval Configuration

The shared backend currently uses these default retrieval settings:

| Parameter | Default | Purpose |
|---|---:|---|
| `top_k` | `3` | Number of final chunks selected for the RAG context |
| `candidate_k` | `10` | Number of semantic retrieval candidates requested |
| `similarity_threshold` | `0.35` | Minimum configured semantic similarity score |

These settings are configurable when constructing `WikiBotBackend`.

The similarity threshold applies to semantic similarity scores; keyword retrieval through BM25 is a separate retrieval mechanism.

## SQLite Storage

The database is located at:

```text
data/wikibot.db
```

The current storage layer manages:

- **Articles:** Cached Wikipedia article metadata and summaries.
- **Chunks:** Processed text chunks used by the RAG pipeline.
- **Links:** Wikipedia links used for knowledge expansion.
- **Article sections:** Ordered article sections used by the Streamlit reader.

SQLite allows previously processed articles to be reused across application sessions. Existing cached articles can also be supplemented with section data when that data is missing.

QnA records will require a backend method and persistent storage implementation, which are planned for a later development milestone.

## Current Development Status

The core RAG workflow is implemented, including Wikipedia data ingestion, text chunking, embeddings, FAISS search, BM25 retrieval, SQLite caching, language model integration, and a shared backend.

The Streamlit interface supports article autocomplete, automatic article loading, collapsible article sections, chatbot interaction with source citations and starter questions, and a separate QnA navigation section.

The QnA UI is currently a work in progress. Saving and fetching QnA records through the backend and SQLite are planned but not yet implemented.

## Development History

1. **Initial RAG Prototype:** Built the initial Wikipedia-based RAG workflow with a command-line interface, allowing questions to be answered using retrieved article content.

2. **Wikipedia Data Processing:** Added Wikipedia article retrieval and processing to prepare article content for downstream retrieval and generation.

3. **Text Cleaning and Chunking:** Introduced text cleaning and chunking to divide article content into manageable passages suitable for embedding and language model context.

4. **Embedding Layer:** Integrated Sentence Transformers using `all-MiniLM-L6-v2` to convert text chunks into vector embeddings.

5. **FAISS Vector Search:** Added FAISS-based semantic retrieval using normalized embeddings and inner-product similarity to identify relevant chunks.

6. **RAG Pipeline:** Introduced a pipeline to combine a user's question, retrieved context, and a language model to generate context-aware answers.

7. **SQLite Knowledge Store:** Added persistent storage for articles, chunks, and Wikipedia links to avoid repeatedly fetching and processing the same content.

8. **Knowledge Manager:** Introduced a knowledge management layer to coordinate article ingestion, caching, and preparation of content for retrieval.

9. **Wikipedia Link Expansion:** Added controlled expansion of knowledge through Wikipedia links, allowing the knowledge base to include related articles.

10. **Multiple LLM Support:** Added selectable language model configurations, including supported causal and sequence-to-sequence models.

11. **CUDA and GPU Support:** Added support for GPU-accelerated model execution when compatible CUDA hardware and dependencies are available.

12. **Streamlit Development Interface:** Introduced a Streamlit interface for selecting models, loading Wikipedia knowledge, and testing RAG question answering interactively.

13. **Sentence-aware Chunking:** Improved chunking to respect sentence boundaries and produce more coherent passages for retrieval and context construction.

14. **FAISS Similarity Scoring and Threshold:** Added semantic similarity scores and a configurable threshold to help filter low-relevance retrieval results.

15. **BM25 Hybrid Retrieval:** Added BM25 keyword search alongside FAISS semantic retrieval to improve coverage for exact terms and keyword-heavy questions.

16. **Shared Backend Architecture:** Refactored wikiBOT to introduce a shared `WikiBotBackend` responsible for managing the core RAG components and their lifecycle. The backend now owns SQLite knowledge storage, Wikipedia knowledge management, embedding and LLM resources, FAISS and BM25 retrieval stores, and RAG pipeline construction, while the CLI and Streamlit layers focus on user interaction. This also enables expensive resources such as the embedding model and LLM to remain loaded and be reused across multiple article loads instead of being recreated by the interface layer.

17. **Wikipedia Article Autocomplete and Auto-Loading:** Added Wikipedia article autocomplete to the Streamlit search interface, allowing users to search and select article titles directly from suggestions while typing. Selecting an article now automatically loads its Wikipedia content into the RAG backend without requiring a separate load action. Suggestions are retrieved through the backend using Wikipedia's search API with short-lived caching to reduce repeated requests, keeping the search and loading logic reusable across different interfaces.

18. **Wikipedia Article Reader:** Added structured Wikipedia article rendering to the Streamlit sidebar. Article sections are stored separately in SQLite and displayed as collapsible sections for easier reading, while the existing chunked representation continues to be used independently for RAG retrieval. The article cache was also extended to populate section data for previously cached articles when needed.

19. **Chatbot and QnA Navigation:** Added a main navigation bar to switch between the chatbot and a dedicated QnA section. Added a plus-button interface beside generated responses and introduced UI helper functions for submitting QnA records containing the article title, question, and answer, as well as fetching records for display. The backend persistence and retrieval implementations remain future work.

20. **Chat Interface Improvements:** Reworked the chat page. Answers now come with the chunks used to produce them, shown in a collapsible "Sources" panel, which required adding `ask_with_sources()` to the pipeline, article context and backend (`ask()` is unchanged, so the CLI is unaffected). Added clickable starter questions, status chips for article, model and device, and Clear and Export controls. Fixed the "Add to Q&A" button, which never fired because clicking it reran the script with no new question and the function returned before the button was rendered; the button now lives in the message history and uses a callback.

## Planned Improvements

- Implement backend methods for saving and fetching QnA records.
- Add persistent QnA storage and connect the UI wrappers to the backend.
- Display saved Q&A records in the QnA section.
- Add QnA management features such as removing records and filtering by article.
- Improve retrieval quality with a dedicated reranking stage.
- Expand retrieval evaluation and testing.
- Improve observability, error handling, and performance measurements.

## License

No license has been specified yet.
