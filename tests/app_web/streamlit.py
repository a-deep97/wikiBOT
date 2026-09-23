import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.backend.manager import WikiBotBackend
from src.models.data import AVAILABLE_MODELS, DEFAULT_MODEL


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="wikiBOT",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# Cached backend
# --------------------------------------------------

@st.cache_resource
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

    # Load shared resources when the app starts.
    backend.load_resources()

    return backend


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("wikiBOT")


article_title = st.sidebar.text_input(
    "Wikipedia Article",
    value="Google"
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
# Initialize backend
# --------------------------------------------------

backend = get_backend(model_key)


# --------------------------------------------------
# Load Knowledge
# --------------------------------------------------

if st.sidebar.button("Load Knowledge"):

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

            st.session_state.model_key = (
                model_key
            )

            # Start a fresh conversation for
            # the newly loaded article.
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

                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )