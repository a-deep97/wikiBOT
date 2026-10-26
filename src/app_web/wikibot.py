
"""Main Streamlit application for wikiBOT."""

from __future__ import annotations

import streamlit as st

from src.app_web.components.article_reader import (
    render_article_reader,
)
from src.app_web.components.chat_interface import (
    render_chat_interface,
)
from src.app_web.components.qna_panel import (
    render_qna_panel,
)
from src.app_web.components.sidebar import (
    render_sidebar,
)
from src.app_web.search_history import (
    add_search,
    load_search_history,
)
from src.app_web.styles import apply_global_styles

from src.backend.wiki_backend import WikiBotBackend
from src.models.data import AVAILABLE_MODELS, DEFAULT_MODEL


# ---------------------------------------------------------
# 1. Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="wikiBOT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# 2. Shared UI styling
# ---------------------------------------------------------

apply_global_styles()



st.markdown(
    """
    <style>
    /* Navigation buttons */
    div[data-testid="stHorizontalBlock"]
    div[data-testid="stButton"] button {
        min-height: 64px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* Active navigation tab: blue */
    div[data-testid="stHorizontalBlock"]
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #2563EB !important;
        color: white !important;
        border: 1px solid #2563EB !important;
    }

    /* Inactive navigation tabs */
    div[data-testid="stHorizontalBlock"]
    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: transparent;
        color: var(--text-color);
        border: 1px solid #D1D5DB;
    }

    /* Hover effect */
    div[data-testid="stHorizontalBlock"]
    div[data-testid="stButton"] button:hover {
        border-color: #2563EB !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 3. Model selection
# ---------------------------------------------------------

def get_model_options() -> list[str]:
    """Return the available model identifiers."""

    if isinstance(AVAILABLE_MODELS, dict):
        return list(AVAILABLE_MODELS.keys())

    return list(AVAILABLE_MODELS)


model_options = get_model_options()

if not model_options:
    st.error("No language models are configured.")
    st.stop()

default_index = (
    model_options.index(DEFAULT_MODEL)
    if DEFAULT_MODEL in model_options
    else 0
)

with st.sidebar:
    st.markdown("# 🧠 wikiBOT")
    st.caption("Your Wikipedia research assistant")
    st.divider()

    model_key = st.selectbox(
        "Language model",
        options=model_options,
        index=default_index,
        key="selected_model",
        help="Choose the language model used to generate answers.",
    )


# ---------------------------------------------------------
# 4. Backend initialization
# ---------------------------------------------------------

def get_backend(model_key: str) -> WikiBotBackend:
    """
    Create a backend instance for the current Streamlit session.

    Keep the backend instance session-specific because it holds
    mutable article-specific state.
    """

    backend = WikiBotBackend(
        model_key=model_key,
        top_k=3,
        candidate_k=10,
        similarity_threshold=0.35,
    )

    backend.load_resources()

    return backend


if (
    "backend" not in st.session_state
    or st.session_state.get("model_key") != model_key
):
    # Create a new session backend only when necessary.
    st.session_state.backend = get_backend(model_key)
    st.session_state.model_key = model_key

    # A model change starts a fresh article/chat context.
    st.session_state.article_title = None
    st.session_state.messages = []

backend = st.session_state.backend


# ---------------------------------------------------------
# 5. Initialize session state
# ---------------------------------------------------------

if "article_title" not in st.session_state:
    st.session_state.article_title = (
        backend.current_article_title
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = load_search_history()


# ---------------------------------------------------------
# 6. Wikipedia autocomplete
# ---------------------------------------------------------

def search_articles(query: str) -> list[str]:
    """Return Wikipedia article suggestions."""

    if not query or not query.strip():
        return []

    backend = st.session_state.get("backend")

    if backend is None:
        return []

    results = backend.suggest_articles(query.strip())

    return results or []



# ---------------------------------------------------------
# 7. Sidebar article search and top navigation
# ---------------------------------------------------------

selected_article = render_sidebar(
    search_function=search_articles,
    recent_searches=st.session_state.recent_searches,
)

st.markdown("## 🧠 wikiBOT")
st.caption("Your Wikipedia research assistant")



# Initialize the active page.
if "active_section" not in st.session_state:
    st.session_state.active_section = "📖 Article Reader"


def set_active_section(section: str) -> None:
    """Update the active navigation section."""
    st.session_state.active_section = section


# Top navigation buttons.
nav1, nav2, nav3 = st.columns(3, gap="medium")

sections = [
    ("📖 Article Reader", nav1, "nav_article_reader"),
    ("💬 Chat", nav2, "nav_chat"),
    ("📚 QnA", nav3, "nav_qna"),
]

for section, column, button_key in sections:
    with column:
        st.button(
            section,
            key=button_key,
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.active_section == section
                else "secondary"
            ),
            on_click=set_active_section,
            args=(section,),
        )

active_section = st.session_state.active_section


# ---------------------------------------------------------
# 8. Load the selected article
# ---------------------------------------------------------

if isinstance(selected_article, str):
    article_to_load = selected_article.strip()
else:
    article_to_load = None

if (
    article_to_load
    and article_to_load
    != st.session_state.get("article_title")
):
    try:
        with st.spinner(
            f"Loading '{article_to_load}' into wikiBOT..."
        ):
            backend.load_knowledge(article_to_load)

        # Update the UI only after the backend has loaded the article.
        loaded_title = backend.current_article_title

        if not loaded_title:
            raise RuntimeError(
                "The backend did not set current_article_title "
                "after loading the article."
            )

        st.session_state.article_title = loaded_title

        # A newly loaded article starts a new conversation.
        st.session_state.messages = []

        # Persist the recent-article history.
        st.session_state.recent_searches = add_search(loaded_title)

        st.toast(
            f"Loaded: {loaded_title}",
            icon="📖",
        )

    except Exception as error:
        st.error(
            f"Could not load '{article_to_load}': {error}"
        )


# ---------------------------------------------------------
# 9. Render the selected page
# ---------------------------------------------------------

article_title = st.session_state.get("article_title")

if active_section == "📖 Article Reader":

    render_article_reader(
        backend=backend,
        article_title=article_title,
    )

elif active_section == "💬 Chat":

    render_chat_interface(
        backend=backend,
        article_title=article_title,
    )

elif active_section == "📚 QnA":

    render_qna_panel(
        backend=backend,
        article_title=article_title,
    )

else:
    st.warning(
        f"Unknown page selected: {active_section}"
    )
