import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.database.sqlite_store import SQLiteStore
from src.knowledge.manager import KnowledgeManager
from src.embeddings.embedder import Embedder
from src.vector.faiss_store import FAISSStore
from src.vector.bm25_store import BM25Store
from src.models.model import Model
from src.models.data import AVAILABLE_MODELS, DEFAULT_MODEL
from src.pipeline import RAGPipeline


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="wikiBOT",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# Cached shared resources
# --------------------------------------------------

@st.cache_resource
def load_embedder():
    print("[EMBEDDER] Loading embedding model...")

    start = time.perf_counter()

    embedder = Embedder()

    print(
        f"[TIMING] Embedder initialization: "
        f"{time.perf_counter() - start:.2f}s"
    )

    return embedder


@st.cache_resource
def load_llm(model_key: str):
    print(f"[LLM] Loading model: {model_key}")

    start = time.perf_counter()

    llm = Model(model_key)

    print(
        f"[TIMING] LLM initialization: "
        f"{time.perf_counter() - start:.2f}s"
    )

    return llm


# --------------------------------------------------
# Article-specific backend
# --------------------------------------------------

@st.cache_resource
def load_backend(
    model_key: str,
    article_title: str
):
    total_start = time.perf_counter()

    # --------------------------------------------------
    # 1. Load SQLite / Wikipedia
    # --------------------------------------------------

    start = time.perf_counter()

    database = SQLiteStore()
    knowledge_manager = KnowledgeManager(database)

    chunks = knowledge_manager.get_article(
        article_title
    )

    if not chunks:
        raise ValueError(
            f"No chunks found for '{article_title}'."
        )

    print(
        f"[TIMING] SQLite/Wikipedia: "
        f"{time.perf_counter() - start:.2f}s"
    )

    # --------------------------------------------------
    # 2. Get cached embedder
    # --------------------------------------------------

    start = time.perf_counter()

    embedder = load_embedder()

    print(
        f"[TIMING] Get embedder: "
        f"{time.perf_counter() - start:.2f}s"
    )

    # --------------------------------------------------
    # 3. Generate embeddings
    # --------------------------------------------------

    start = time.perf_counter()

    chunk_texts = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = embedder.embed_documents(
        chunk_texts
    )

    print(
        f"[TIMING] Embedding generation "
        f"({len(chunks)} chunks): "
        f"{time.perf_counter() - start:.2f}s"
    )

    # --------------------------------------------------
    # 4. Build FAISS
    # --------------------------------------------------

    start = time.perf_counter()

    vector_store = FAISSStore(
        dimension=embeddings.shape[1]
    )

    vector_store.add(
        embeddings,
        chunks
    )

    print(
        f"[TIMING] FAISS: "
        f"{time.perf_counter() - start:.2f}s"
    )

    # --------------------------------------------------
    # 5. Build BM25
    # --------------------------------------------------

    start = time.perf_counter()

    bm25_store = BM25Store()
    bm25_store.build(chunks)

    print(
        f"[TIMING] BM25: "
        f"{time.perf_counter() - start:.2f}s"
    )

    # --------------------------------------------------
    # 6. Get cached LLM
    # --------------------------------------------------

    start = time.perf_counter()

    llm = load_llm(model_key)

    print(
        f"[TIMING] Get LLM: "
        f"{time.perf_counter() - start:.2f}s"
    )

    # --------------------------------------------------
    # 7. Build RAG pipeline
    # --------------------------------------------------

    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        bm25_store=bm25_store,
        llm=llm,
        top_k=3,
        candidate_k=10,
        similarity_threshold=0.35
    )

    print(
        f"[TIMING] TOTAL backend load: "
        f"{time.perf_counter() - total_start:.2f}s"
    )

    return pipeline


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("wikiBOT")

article_title = st.sidebar.text_input(
    "Wikipedia Article",
    value="Python"
)

model_keys = list(AVAILABLE_MODELS.keys())

model_key = st.sidebar.selectbox(
    "Model",
    model_keys,
    index=model_keys.index(DEFAULT_MODEL),
    format_func=lambda key: (
        f"{key} - {AVAILABLE_MODELS[key]['name']}"
    )
)


# --------------------------------------------------
# Preload shared resources
# --------------------------------------------------

# The embedding model is independent of the
# article and model selection, so load it once.
load_embedder()

# Load the currently selected LLM.
# @st.cache_resource ensures it is loaded only once
# for each model key.
load_llm(model_key)


# --------------------------------------------------
# Load Knowledge button
# --------------------------------------------------

if st.sidebar.button("Load Knowledge"):

    with st.spinner(
        f"Loading '{article_title}'..."
    ):
        try:
            pipeline = load_backend(
                model_key=model_key,
                article_title=article_title
            )

            st.session_state.pipeline = pipeline
            st.session_state.article_title = article_title
            st.session_state.model_key = model_key

            # Start a fresh conversation when
            # a new article/model is loaded.
            st.session_state.messages = []

            st.sidebar.success(
                f"Loaded '{article_title}'"
            )

        except Exception as error:
            st.sidebar.error(
                f"Failed to load knowledge: {error}"
            )


# --------------------------------------------------
# Main UI
# --------------------------------------------------

st.title("🤖 wikiBOT")

st.write(
    "Ask questions about a Wikipedia article "
    "using local RAG."
)


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question..."
)


if question:

    if "pipeline" not in st.session_state:

        st.warning(
            "Please load a Wikipedia article first."
        )

    else:

        # Display user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:
                    answer = (
                        st.session_state.pipeline
                        .ask(question)
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as error:

                    error_message = (
                        f"Error generating answer: {error}"
                    )

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )