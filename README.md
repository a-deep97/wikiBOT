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
- **Hybrid retrieval:** Combine FAISS semantic-search results and BM25 keyword-search results using Reciprocal Rank Fusion (RRF).
- **Similarity filtering:** Filter FAISS semantic-search results using a configurable similarity threshold before hybrid ranking.
- **Rank-based fusion:** Combine retriever rankings without directly comparing their raw scores, improving the selection of relevant context chunks.
- **Persistent storage:** Cache article content, chunks, links, sections, and saved Q&A in SQLite.
- **Knowledge management:** Reuse cached article data and expand knowledge through Wikipedia links.
- **Multiple language models:** Select from supported causal and sequence-to-sequence models.
- **GPU support:** Support CUDA-enabled model execution when compatible hardware and dependencies are available.
- **Shared backend:** Centralize model resources, storage, retrieval, and RAG orchestration.
- **Streamlit interface:** Search for Wikipedia articles using autocomplete, automatically load selected articles, read article sections, and interact with the chatbot.
- **Source-backed answers:** Every chat answer includes a collapsible "Sources" panel showing the article sections and text snippets that were used as context.
- **Chat conveniences:** Clickable starter questions, a status bar showing the article, model and device, and Clear and Export (Markdown) controls.
- **Article reader:** Browse the loaded article with summary stats (sections, words, search chunks), a link to the original Wikipedia page, a section filter, an "Expand all" toggle, and an "Ask about this section" button that jumps to the chat with a ready-made question.
- **Saved Q&A:** Save question-and-answer pairs from the chat with one click. Records are stored in SQLite per article and loaded again whenever that article is opened.
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

The application uses a shared `WikiBotBackend` to coordinate article loading, storage, retrieval, and question answering.

For each question, the RAG pipeline retrieves candidates using FAISS semantic search and BM25 keyword search. FAISS results are filtered using the configured similarity threshold, and the remaining results are combined with BM25 results using Reciprocal Rank Fusion (RRF). The highest-ranked unique chunks are selected as context for the language model, which generates the final answer.

## Project Structure

```text
wikiBOT/
├── src/
│   ├── cli.py
│   ├── config.py
│   ├── pipeline.py
│   ├── backend/
│   │   ├── wiki_backend.py
│   │   ├── article_context.py
│   │   └── qna_manager.py
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
│   │   ├── sqlite_store.py
│   │   └── qna_store.py
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
| `src/pipeline.py` | RAG pipeline orchestration; combines FAISS and BM25 rankings using RRF, retrieves context, and provides `ask()` and `ask_with_sources()` methods |
| `src/backend/wiki_backend.py` | Shared backend: resource lifecycle, article loading, autocomplete, question answering, and access to saved Q&A through `backend.qna` |
| `src/backend/qna_manager.py` | Saves, retrieves, and deletes Q&A records in SQLite using the shared database connection. Prevents duplicate records and supports deletion by unique record ID. |

| `src/backend/article_context.py` | Retrieval stores and pipeline for one loaded article, including its chunk count |
| `src/app_web/wikibot.py` | Streamlit entry point, model selection, and article loading |
| `src/app_web/components/` | Streamlit UI sections: sidebar, search, article reader, chat, and Q&A |
| `src/app_web/helpers.py` | Reusable UI helpers: section rendering with filter and expand options, word counts, the section-to-chat jump, chat source cleanup, status chips, and Q&A UI wrappers |
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
| `src/database/sqlite_store.py` | Persistent SQLite storage and retrieval for articles, chunks, links, and sections |
| `src/database/qna_store.py` | Saved question-and-answer records, stored in the same SQLite database through a shared connection |
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

When a question is submitted, wikiBOT retrieves relevant text chunks using hybrid retrieval, combining semantic search and keyword search.

- **FAISS semantic retrieval:** Uses text embeddings to find chunks that are semantically similar to the question.
- **BM25 keyword retrieval:** Finds chunks with relevant keyword matches, helping with exact terms and names.
- **Similarity filtering:** Applies the configured similarity threshold to FAISS results.
- **Reciprocal Rank Fusion (RRF):** Combines the ranked FAISS and BM25 results into a unified ranking.
- **Context selection:** Selects the top-ranked chunks to provide relevant context to the language model.

This approach combines semantic relevance with keyword matching to improve retrieval coverage.


### 4. Generate an answer

The RAG pipeline combines the user's question with the retrieved context and sends the resulting prompt to the selected language model.

The model generates an answer based on the available context. The chat calls `ask_with_sources()`, so each answer comes back together with the chunks that were used to build its prompt.

### Chat features

- **Sources:** Each answer has a collapsible "Sources" panel listing the article section and a short snippet for every retrieved chunk. The "Article:" and "Section:" header lines added during chunking are stripped for display.
- **Starter questions:** When a conversation is empty, the chat suggests a summary question plus questions based on the article's longest sections. Clicking one submits it.
- **Status bar:** Small chips show the loaded article, the selected model, and whether it runs on GPU or CPU.
- **Clear and Export:** Clear starts a fresh conversation; Export downloads the conversation as Markdown, including the source sections for each answer.
- **Add to Q&A:** Each answer has an "Add to Q&A" button that saves the question and answer for the current article. The button switches to a "Saved" state, and saving the same pair again reports that it is already saved instead of creating a duplicate.

### 5. Read the source article

The **Article Reader** page shows the loaded article's sections in collapsible panels. Article sections are stored separately from retrieval chunks to support both reading and question answering.

- **Overview:** Metrics show the number of sections, the approximate word count, and how many chunks the article was split into for retrieval. An "Open on Wikipedia" button links to the original article.
- **Filter:** Type in the filter box to show only sections whose title or text contains the search term. The filter is remembered per article.
- **Expand all:** A toggle opens or collapses every section at once. The Introduction opens by default when no filter is active.
- **Ask about this section:** Each section has a button that switches to the chat and submits a question about it, for example "What does the article say about Philosophy (Design)?". The Introduction maps to a summary question.


### 6. Save, review, and delete Q&A records

Use **Add to Q&A** under a chat answer to save the question and answer for the current article. The UI helper `add_qna` passes the record to `QnAManager`, accessible through `backend.qna`, which stores it in the `qna` table of the SQLite database.

The **Q&A** page lists saved records for the current article, newest first. Records persist across application restarts because they are stored in SQLite.

**Available actions:**

- **Save:** Save a question and its answer for the current article.
- **Duplicate prevention:** Saving an identical question and answer for the same article does not create another record.
- **Review:** Expand a saved record to view its answer.
- **Delete:** Select a record's Delete button to request deletion.
- **Deletion confirmation:** Confirm the action to permanently delete the selected record, or cancel to keep it.
- **Automatic refresh:** After deletion, the Q&A list refreshes to reflect the database changes.

Each record has a unique database ID. Deletion uses this ID to target the selected record without deleting other saved Q&A records.


## Retrieval Configuration

The shared backend uses the following retrieval parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `top_k` | `3` | Number of final chunks selected for the RAG context |
| `candidate_k` | `10` | Number of candidate results requested from each retriever before final selection |
| `similarity_threshold` | `0.35` | Minimum configured FAISS semantic similarity score |

These settings are configurable when constructing `WikiBotBackend`.

The similarity threshold applies only to FAISS semantic-search results. BM25 uses keyword-based relevance scoring, and its raw scores are not directly compared with FAISS scores.

RRF combines the retrievers using their result rankings. The current implementation uses a rank constant of `60` to calculate reciprocal-rank contributions.

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
- **Q&A:** Question-and-answer pairs saved from the chat, with the article title and the time they were saved. This table is managed by `QnAManager` in `src/backend/qna_manager.py`, which shares the main store's SQLite connection and creates the table automatically if it is missing.

SQLite allows previously processed articles to be reused across application sessions. Existing cached articles can also be supplemented with section data when that data is missing.


## Current Development Status

The core RAG workflow is implemented, including Wikipedia data ingestion, text chunking, embeddings, FAISS search, BM25 retrieval, SQLite caching, language model integration, and a shared backend.

The Streamlit interface supports article autocomplete, automatic article loading, an article reader with filtering and per-section questions, chatbot interaction with source citations and starter questions, and a separate QnA navigation section.

Saved Q&A is implemented end to end: answers can be saved from the chat, are stored in SQLite, and are listed on the Q&A page.

Saved Q&A management supports saving, listing, duplicate prevention, and deletion. Users can delete individual saved records from the Q&A page after confirmation. Deletions are committed to SQLite and remain effective after restarting the application.

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

15. **Hybrid Retrieval and RRF Ranking:** Added BM25 keyword search alongside FAISS semantic retrieval, then combined the ranked results using Reciprocal Rank Fusion (RRF). FAISS results are filtered using a configurable similarity threshold, and RRF combines the rankings to select relevant context chunks without directly comparing raw scores.

16. **Shared Backend Architecture:** Refactored wikiBOT to introduce a shared `WikiBotBackend` responsible for managing the core RAG components and their lifecycle. The backend now owns SQLite knowledge storage, Wikipedia knowledge management, embedding and LLM resources, FAISS and BM25 retrieval stores, and RAG pipeline construction, while the CLI and Streamlit layers focus on user interaction. This also enables expensive resources such as the embedding model and LLM to remain loaded and be reused across multiple article loads instead of being recreated by the interface layer.

17. **Wikipedia Article Autocomplete and Auto-Loading:** Added Wikipedia article autocomplete to the Streamlit search interface, allowing users to search and select article titles directly from suggestions while typing. Selecting an article now automatically loads its Wikipedia content into the RAG backend without requiring a separate load action. Suggestions are retrieved through the backend using Wikipedia's search API with short-lived caching to reduce repeated requests, keeping the search and loading logic reusable across different interfaces.

18. **Wikipedia Article Reader:** Added structured Wikipedia article rendering to the Streamlit sidebar. Article sections are stored separately in SQLite and displayed as collapsible sections for easier reading, while the existing chunked representation continues to be used independently for RAG retrieval. The article cache was also extended to populate section data for previously cached articles when needed.

19. **Chatbot and QnA Navigation:** Added a main navigation bar to switch between the chatbot and a dedicated QnA section. Added a plus-button interface beside generated responses and introduced UI helper functions for submitting QnA records containing the article title, question, and answer, as well as fetching records for display. The backend persistence and retrieval implementations remain future work.

20. **Chat Interface Improvements:** Reworked the chat page. Answers now come with the chunks used to produce them, shown in a collapsible "Sources" panel, which required adding `ask_with_sources()` to the pipeline, article context and backend (`ask()` is unchanged, so the CLI is unaffected). Added clickable starter questions, status chips for article, model and device, and Clear and Export controls. Fixed the "Add to Q&A" button, which never fired because clicking it reran the script with no new question and the function returned before the button was rendered; the button now lives in the message history and uses a callback.

21. **Article Reader Improvements:** Reworked the Article Reader page. Added summary stats (sections, words, retrieval chunks), an "Open on Wikipedia" link, a section filter, an "Expand all" toggle, and an "Ask about this section" button on each section. The button uses a callback that sets the chat's pending question and switches the navigation to the chat page, so the question is answered immediately. Added `chunk_count` to `ArticleContext` and a `current_chunk_count` property on `WikiBotBackend` to support the stats.

22. **Persistent Saved Q&A:** Implemented end-to-end saving and retrieval of question-and-answer records. `QnAManager` manages Q&A records in SQLite using the shared database connection. Answers can be saved from the chat, duplicate records are handled, and saved Q&A records can be reviewed for the selected article across application sessions.

23. **Q&A Deletion:** Extended saved Q&A management to support deleting individual question-and-answer records from the Streamlit interface. Users must confirm deletion before the selected record is permanently removed from SQLite. Other saved records remain unaffected.


## Planned Improvements

- Add Q&A management features such as removing records, searching, and exporting.
- Add a view of saved Q&A across all articles.
- Store the source chunks with saved Q&A so the sources can be shown on the Q&A page.

## License

No license has been specified yet.