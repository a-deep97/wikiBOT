"""Chat interface for asking questions about the loaded article."""

import re

import streamlit as st

from src.app_web.helpers import (
    add_qna,
    chunk_body,
    render_status_bar,
    section_question,
)
from src.app_web.styles import render_page_header

NO_ANSWER_TEXT = "I couldn't generate an answer for that question."
SNIPPET_LENGTH = 400


# ---------------------------------------------------------
# Callbacks (run before the script reruns)
# ---------------------------------------------------------

def _queue_question(question: str) -> None:
    """Submit a suggested question on the next run."""

    st.session_state.pending_question = question


def _clear_chat() -> None:
    st.session_state.messages = []


def _save_to_qna(backend, article_title: str, index: int) -> None:
    """Save one answered question to the Q&A records."""

    message = st.session_state.messages[index]

    try:
        created = add_qna(
            backend=backend,
            title=article_title,
            question=message["question"],
            answer=message["content"],
        )
    except Exception as error:
        st.toast(f"Could not save Q&A: {error}", icon="⚠️")
        return

    message["saved"] = True
    st.toast(
        "Saved to Q&A" if created else "Already in your Q&A",
        icon="📚",
    )


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _starter_questions(
    backend,
    article_title: str,
    limit: int = 4,
) -> list[str]:
    """Suggest questions: a summary plus the article's meatiest sections."""

    questions = [f"Give me a short summary of {article_title}."]

    try:
        sections = backend.get_article_sections()
    except Exception:
        return questions

    candidates = [
        (index, section)
        for index, section in enumerate(sections)
        if section["title"].strip().lower() != "introduction"
        and len(section["text"].split()) >= 60
    ]

    # Take the longest sections, then restore article order.
    longest = sorted(
        candidates,
        key=lambda item: len(item[1]["text"]),
        reverse=True,
    )[: limit - 1]

    longest.sort(key=lambda item: item[0])

    questions.extend(
        section_question(article_title, section["title"])
        for _, section in longest
    )

    return questions


def _snippet(text: str) -> str:
    if len(text) <= SNIPPET_LENGTH:
        return text

    return text[:SNIPPET_LENGTH].rsplit(" ", 1)[0] + "…"


def _source_labels(sources: list[dict]) -> list[str]:
    """Unique section names, in the order they were retrieved."""

    labels = []

    for source in sources:
        label = source.get("section") or "Article"

        if label not in labels:
            labels.append(label)

    return labels


def _build_transcript(article_title: str, messages: list[dict]) -> str:
    lines = [f"# wikiBOT conversation: {article_title}", ""]

    for message in messages:
        if message["role"] == "user":
            lines += [f"**Q:** {message['content']}", ""]
            continue

        lines += [f"**A:** {message['content']}", ""]

        labels = _source_labels(message.get("sources") or [])

        if labels:
            lines += [f"_Sources: {', '.join(labels)}_", ""]

    return "\n".join(lines)


def _filename(article_title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", article_title.lower()).strip("-")

    return f"wikibot-{slug or 'article'}.md"


# ---------------------------------------------------------
# Rendering
# ---------------------------------------------------------

def _render_toolbar(article_title: str, messages: list[dict]) -> None:
    _, clear_column, export_column = st.columns([6, 1.4, 1.8])

    with clear_column:
        st.button(
            "🗑️ Clear",
            key="clear_chat",
            on_click=_clear_chat,
            use_container_width=True,
            help="Start a fresh conversation about this article.",
        )

    with export_column:
        st.download_button(
            "⬇️ Export",
            data=_build_transcript(article_title, messages),
            file_name=_filename(article_title),
            mime="text/markdown",
            key="export_chat",
            use_container_width=True,
            help="Download this conversation as Markdown.",
        )


def _render_starters(backend, article_title: str) -> None:
    st.markdown(
        """
        <div class="wikibot-card">
            <h3>💡 Start exploring</h3>
            <p>
                Ask anything about this article, or try one of
                these questions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    questions = _starter_questions(backend, article_title)
    columns = st.columns(2)

    for index, question in enumerate(questions):
        with columns[index % 2]:
            st.button(
                question,
                key=f"starter_{index}",
                on_click=_queue_question,
                args=(question,),
                use_container_width=True,
            )


def _render_message(
    backend,
    article_title: str,
    index: int,
    message: dict,
) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] != "assistant":
            return

        sources = message.get("sources") or []

        if sources:
            with st.expander(f"📎 Sources ({len(sources)})"):
                for number, source in enumerate(sources, start=1):
                    st.markdown(
                        f"**{number}. {source.get('section') or 'Article'}**"
                    )
                    st.caption(_snippet(source["text"]))

        saved = message.get("saved", False)

        st.button(
            "📚 Saved to Q&A" if saved else "➕ Add to Q&A",
            key=f"save_qna_{index}",
            disabled=saved,
            on_click=_save_to_qna,
            args=(backend, article_title, index),
        )


def _answer_question(backend, question: str) -> bool:
    """Generate an answer and append it to the conversation."""

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching the article and preparing an answer..."
            ):
                result = backend.ask_with_sources(question)
        except Exception as error:
            st.error(f"Unable to answer the question: {error}")
            return False

    answer = result.get("answer")

    if not answer:
        answer = NO_ANSWER_TEXT
    elif not isinstance(answer, str):
        answer = str(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "question": question,
            "saved": False,
            "sources": [
                {
                    "section": source.get("section", ""),
                    "text": chunk_body(source.get("content", "")),
                }
                for source in result.get("sources", [])
            ],
        }
    )

    return True


def render_chat_interface(
    backend,
    article_title: str | None,
) -> None:
    """Render the chat UI and process submitted questions."""

    render_page_header(
        title="Chat with wikiBOT",
        subtitle=(
            "Ask questions about the article loaded into your "
            "current knowledge base."
        ),
        eyebrow="RESEARCH ASSISTANT",
    )

    if not backend or not article_title:
        st.info(
            "Load a Wikipedia article from the sidebar before "
            "starting a conversation."
        )
        return

    render_status_bar(backend, article_title)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    messages = st.session_state.messages

    # A question can come from the chat box, a suggestion button, or
    # the "Ask about this section" button in the article reader.
    typed_question = st.chat_input(
        f"Ask about {article_title}...",
        key="article_chat_input",
    )

    question = st.session_state.pop("pending_question", None) or typed_question

    if question:
        messages.append({"role": "user", "content": question})

    if messages:
        _render_toolbar(article_title, messages)
    else:
        _render_starters(backend, article_title)

    for index, message in enumerate(messages):
        _render_message(backend, article_title, index, message)

    if question and _answer_question(backend, question):
        # Rerun so the new answer is drawn by the same code as the
        # history, including its sources and the "Add to Q&A" button.
        st.rerun()
