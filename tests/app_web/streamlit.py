import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from streamlit_searchbox import st_searchbox

from src.backend.manager import WikiBotBackend
from src.models.data import AVAILABLE_MODELS, DEFAULT_MODEL
from tests.app_web.helpers import render_article_sections

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="wikiBOT",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# Sidebar sizing
# --------------------------------------------------
#
# Streamlit's sidebar is natively resizable.
#
# Default width:
#   ~43% of the main usable application width
#
# This gives approximately:
#
#   Article : Chatbot
#      3   :    4
#
# The sidebar itself remains draggable.
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* Default article/sidebar width */
    [data-testid="stSidebar"] {
        min-width: 360px !important;
        width: 51% !important;
        max-width: 800px !important;
    }

    /* Keep the sidebar contents inside the available width */
    [data-testid="stSidebarContent"] {
        min-width: 0 !important;
    }

    /* Prevent the main chatbot area from becoming too narrow */
    [data-testid="stMain"] {
        min-width: 600px !important;
    }

    </style>

    <style>
    [data-testid="stSpinner"] {
        font-size: 20px !important;
    }

    [data-testid="stSpinner"] p {
        font-size: 30px !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Cached backend
# --------------------------------------------------

@st.cache_resource(show_spinner=" Waking up wikiBOT... Loading its brain.")
def get_backend(model_key: str):
    """
    Create and cache a wikiBOT backend for a model.
    """

    backend = WikiBotBackend(
        model_key=model_key,
        top_k=3,
        candidate_k=10,
        similarity_threshold=0.35
    )

    backend.load_resources()

    return backend


# --------------------------------------------------
# Initialize backend
# --------------------------------------------------

model_keys = list(AVAILABLE_MODELS.keys())

model_key = st.sidebar.selectbox(
    "Model",
    model_keys,
    index=model_keys.index(DEFAULT_MODEL),
    format_func=lambda key: (
        f"{key} - {AVAILABLE_MODELS[key]['name']}"
    )
)

backend = get_backend(model_key)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("wikiBOT")

    # ----------------------------------------------
    # Article search
    # ----------------------------------------------

    def search_articles(
        query: str
    ) -> list[str]:
        """Get article suggestions from the backend."""

        if len(query.strip()) < 2:
            return []

        try:
            return backend.suggest_articles(
                query=query,
                limit=8
            )

        except Exception as error:

            print(
                f"[SEARCH] Suggestion error: "
                f"{error}"
            )

            return []


    article_title = st_searchbox(
        search_function=search_articles,
        placeholder="Search Wikipedia articles...",
        label="Wikipedia Article",
        key="wiki_article_search",
        debounce=300,
        edit_after_submit="option",
    )

    # ----------------------------------------------
    # Load article
    # ----------------------------------------------

    if article_title:

        article_title = article_title.strip()

        if (
            article_title
            and article_title
            != st.session_state.get("article_title")
        ):

            with st.spinner(
                f"Loading '{article_title}'..."
            ):

                try:

                    backend.load_knowledge(
                        article_title
                    )

                    st.session_state.backend = (
                        backend
                    )

                    st.session_state.article_title = (
                        article_title
                    )

                    st.session_state.model_key = (
                        model_key
                    )

                    st.session_state.messages = []

                    st.success(
                        f"Loaded '{article_title}'"
                    )

                except Exception as error:

                    st.error(
                        f"Failed to load knowledge: "
                        f"{error}"
                    )

    # ----------------------------------------------
    # Article panel
    # ----------------------------------------------

    st.divider()

    st.subheader("Article")

    if st.session_state.get("article_title"):

        # Inner scrollable article panel.
        with st.container(
            height=650,
            border=True
        ):

            render_article_sections(
                backend
            )

    else:

        st.caption(
            "Search and select a Wikipedia article "
            "to view its contents."
        )


# --------------------------------------------------
# Main chatbot area
# --------------------------------------------------

st.title("🤖 wikiBOT")

st.write(
    "Ask questions about a Wikipedia article "
    "using local RAG."
)


# --------------------------------------------------
# Current article
# --------------------------------------------------

if st.session_state.get("article_title"):

    st.caption(
        f"Article: "
        f"{st.session_state['article_title']}"
    )


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question..."
)


if question:

    if "backend" not in st.session_state:

        st.warning(
            "Please load a Wikipedia article first."
        )

    else:

        # ------------------------------------------
        # Display user question
        # ------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # ------------------------------------------
        # Generate answer
        # ------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer = (
                        st.session_state.backend
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
                        f"Error generating answer: "
                        f"{error}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )