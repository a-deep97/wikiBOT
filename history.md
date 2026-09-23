# Development History

### 1. Initial RAG Prototype

Started the project as a basic Wikipedia-based RAG application with a CLI interface. Implemented the fundamental pipeline of fetching Wikipedia content, processing it into chunks, generating embeddings, performing vector search, and passing retrieved context to a local LLM. The initial implementation intentionally avoided frameworks such as LangChain to understand the underlying RAG components.

### 2. Wikipedia Data Processing

Added a dedicated Wikipedia data layer to fetch articles, validate pages, extract summaries and sections, filter unnecessary sections such as references and external links, and collect links to related Wikipedia articles. Structured the fetched data into article, section, text, and link information.

### 3. Text Cleaning and Chunking

Added text preprocessing and section-aware chunking with configurable chunk size and overlap. Each chunk was given article, section, and chunk-index metadata so that retrieved content could retain information about its original location within the Wikipedia article.

### 4. Embedding Layer

Introduced a dedicated embedding component using Sentence Transformers and `all-MiniLM-L6-v2`. Added separate functionality for embedding individual queries and collections of documents, producing 384-dimensional vector representations for semantic retrieval.

### 5. FAISS Vector Search

Added FAISS as the vector search layer and connected vector entries with their original document chunks. Implemented semantic retrieval using normalized vectors with inner product, effectively providing cosine-similarity search over the embedded knowledge.

### 6. RAG Pipeline

Introduced a dedicated `RAGPipeline` to separate retrieval, prompt construction, and LLM generation. The pipeline embeds the user's question, retrieves the most relevant chunks from FAISS, constructs a context-based prompt, and generates an answer using the local LLM with instructions to rely only on the retrieved context.

### 7. SQLite Knowledge Store

Added a persistent SQLite database to avoid repeatedly downloading and processing the same Wikipedia content. Introduced separate storage for articles, processed chunks, and links between Wikipedia articles, creating a persistent local knowledge base.

### 8. Knowledge Manager

Introduced `KnowledgeManager` to coordinate Wikipedia fetching and SQLite caching. Article requests now check the local database first and fetch, process, and store the article only when it is not already available, separating knowledge management from both the CLI and RAG pipeline.

### 9. Wikipedia Link Expansion

Extended the knowledge layer to preserve relationships between Wikipedia articles through stored links. Designed controlled expansion so that missing linked articles can be discovered and added to the local knowledge base without recursively crawling an uncontrolled portion of Wikipedia.

### 10. Multiple LLM Support

Separated model configuration from model implementation and introduced selectable model keys. Added support for both causal and sequence-to-sequence architectures, allowing models such as Qwen, FLAN-T5, and Mistral to be configured and selected without changing the RAG pipeline.

### 11. CUDA and GPU Support

Added automatic hardware detection for PyTorch CUDA support, GPU inference when CUDA is available, and CPU fallback with warnings when it is not. Verified local GPU inference using an NVIDIA RTX 3050 and added device and GPU information during model initialization.

### 12. Sentence-aware Chunking

Updated the text chunking process to preserve sentence boundaries and maintain sentence-level overlap, preventing chunks from cutting through sentences and improving the quality of retrieved context.

### 13. FAISS Similarity Scoring and Threshold

Updated FAISS retrieval to return cosine similarity scores and added a configurable similarity threshold to filter out low-relevance results before they are passed to the RAG pipeline.

### 14. BM25 Hybrid Retrieval

Added BM25 keyword-based retrieval alongside FAISS semantic search to improve retrieval for exact terms, names, numbers, versions, and technical keywords. The pipeline now combines candidates from both retrieval methods before passing the final context to the LLM.
