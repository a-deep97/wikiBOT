import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.backend.manager import WikiBotBackend
from src.models.data import AVAILABLE_MODELS, DEFAULT_MODEL
from tests.app_web.helpers import (
    add_qna,
    get_qna,
    render_article_sections,
)
from streamlit_searchbox import st_searchbox


st.set_page_config(
    page_title="wikiBOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Global UI styling
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       Hide Streamlit top toolbar
       ------------------------------------------------------- */

    [data-testid="stToolbar"] {
        display: none;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* -------------------------------------------------------
       Main navigation
       ------------------------------------------------------- */

    div[data-testid="stRadio"] > div {
        gap: 8px;
    }

    div[data-testid="stRadio"] label {
        font-size: 20px !important;
        font-weight: 600 !important;
    }


    /* -------------------------------------------------------
       QnA add button
       ------------------------------------------------------- */

    button[key^="add_qna_"] {
        font-size: 22px !important;
        font-weight: 600 !important;
        min-height: 38px !important;
        width: 38px !important;
        padding: 0 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Backend
# ============================================================

@st.cache_resource(
    show_spinner="🧠 Waking up wikiBOT... Loading its brain."
)
def get_backend(model_key: str):
    backend = WikiBotBackend(
        model_key=model_key,
        top_k=3,
        candidate_k=10,
        similarity_threshold=0.35,
    )

    backend.load_resources()

    return backend


# ============================================================
# Model selection
# ============================================================

model_keys = list(AVAILABLE_MODELS.keys())

model_key = st.sidebar.selectbox(
    "Model",
    model_keys,
    index=model_keys.index(DEFAULT_MODEL),
    format_func=lambda key: (
        f"{key} - {AVAILABLE_MODELS[key]['name']}"
    ),
)


backend = get_backend(model_key)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("wikiBOT")


def search_articles(query: str) -> list[str]:
    """Return Wikipedia article suggestions."""

    if len(query.strip()) < 2:
        return []

    try:
        with st.spinner("🔎 Searching Wikipedia..."):
            return backend.suggest_articles(
                query=query,
                limit=8,
            )

    except Exception as error:
        print(f"[SEARCH] Suggestion error: {error}")
        return []


with st.sidebar:

    # --------------------------------------------------------
    # Wikipedia search
    # --------------------------------------------------------

    article_title = st_searchbox(
        search_function=search_articles,
        placeholder="Search Wikipedia articles...",
        label=None,
        key="wiki_article_search_v2",
        debounce=300,
        edit_after_submit="option",
        default=None,
        default_searchterm="",
    )


    # --------------------------------------------------------
    # Load article
    # --------------------------------------------------------

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

                    st.session_state.backend = backend
                    st.session_state.article_title = (
                        article_title
                    )
                    st.session_state.model_key = model_key
                    st.session_state.messages = []

                    st.success(
                        f"Loaded '{article_title}'"
                    )

                except Exception as error:

                    st.error(
                        f"Failed to load knowledge: {error}"
                    )


    st.divider()


    # --------------------------------------------------------
    # Article reader
    # --------------------------------------------------------

    st.subheader("Article")

    if st.session_state.get("article_title"):

        with st.container(
            height=650,
            border=True,
        ):
            render_article_sections(
                backend
            )

    else:

        st.caption(
            "Search and select a Wikipedia article "
            "to view its contents."
        )


# ============================================================
# Main navigation
# ============================================================

if "active_section" not in st.session_state:
    st.session_state.active_section = "🤖 Chatbot"


active_section = st.radio(
    "Navigation",
    ["🤖 Chatbot", "📚 QnA"],
    horizontal=True,
    label_visibility="collapsed",
    key="active_section",
)


# ============================================================
# Chatbot
# ============================================================

if active_section == "🤖 Chatbot":

    st.title("🤖 wikiBOT")

    st.write(
        "Ask questions about a Wikipedia article "
        "using local RAG."
    )


    # --------------------------------------------------------
    # Initialize chat history
    # --------------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []


    # --------------------------------------------------------
    # Render chat history
    # --------------------------------------------------------

    for index, message in enumerate(st.session_state.messages):

        with st.chat_message(message["role"]):

            if message["role"] == "user":
                st.markdown(message["content"])

            else:
                col_answer, col_add = st.columns(
                    [0.90, 0.10],
                    vertical_alignment="top",
                )

                with col_answer:
                    st.markdown(message["content"])

                with col_add:
                    if st.button(
                        "➕",
                        key=f"add_qna_{index}",
                        help="Add this Q&A to QnA",
                    ):
                        question = ""

                        if index > 0:
                            previous_message = st.session_state.messages[index - 1]

                            if previous_message["role"] == "user":
                                question = previous_message["content"]

                        title = st.session_state.get("article_title", "")

                        if question and title:
                            add_qna(
                                backend=st.session_state.backend,
                                title=title,
                                question=question,
                                answer=message["content"],
                            )

                            st.toast("Added to QnA ✓")


# ============================================================
# QnA
# ============================================================

elif active_section == "📚 QnA":
    st.title("📚 QnA")

    title = st.session_state.get("article_title")

    qna_records = get_qna(
        backend=st.session_state.backend,
        title=title,
    )

    if not qna_records:
        st.info("No saved Q&A yet.")
    else:
        for record in qna_records:
            st.subheader(record["question"])
            st.write(record["answer"])
            st.divider()


# ============================================================
# Chat input
#
# IMPORTANT:
# Keep this outside st.tabs() / navigation containers.
# This allows Streamlit to keep the input at the bottom.
# ============================================================

if active_section == "🤖 Chatbot":

    question = st.chat_input(
        "Ask a question..."
    )

    if question:

        # ----------------------------------------------------
        # Check whether an article has been loaded
        # ----------------------------------------------------

        if (
            "backend" not in st.session_state
            or not backend.current_article_title
        ):

            st.warning(
                "Please load a Wikipedia article first."
            )

        else:

            # ------------------------------------------------
            # Save user message
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )


            with st.chat_message("user"):
                st.markdown(question)


            # ------------------------------------------------
            # Generate answer
            # ------------------------------------------------

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = st.session_state.backend.ask(question)

                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )

                        col_answer, col_add = st.columns(
                            [0.90, 0.10],
                            vertical_alignment="top",
                        )

                        with col_answer:
                            st.markdown(answer)

                        with col_add:
                            st.button(
                                "➕",
                                key=f"add_qna_{len(st.session_state.messages) - 1}",
                                help="Add this Q&A to QnA",
                            )

                    except Exception as error:
                        error_message = f"Error generating answer: {error}"

                        st.error(error_message)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_message,
                            }
                        )