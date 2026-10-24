"""Saved question-and-answer UI."""

import streamlit as st

from src.app_web.helpers import get_qna
from src.app_web.styles import render_page_header


def render_qna_panel(
    backend,
    article_title: str | None,
) -> None:
    """Display saved Q&A for the selected article."""

    render_page_header(
        title="Saved Q&A",
        subtitle="Review previously saved questions and answers.",
        eyebrow="KNOWLEDGE HISTORY",
    )

    if not backend or not article_title:
        st.info(
            "Load a Wikipedia article before viewing its saved Q&A."
        )
        return

    st.caption(f"Current article: **{article_title}**")

    try:
        qna_records = get_qna(
            backend=backend,
            title=article_title,
        )
    except Exception as error:
        st.error(f"Unable to load saved Q&A: {error}")
        return

    if not qna_records:
        st.markdown(
            """
            <div class="wikibot-card">
                <h3>💬 No saved questions yet</h3>
                <p>
                    Your saved questions and answers will appear
                    here when they are available for this article.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.caption(f"{len(qna_records)} saved question(s)")

    for index, record in enumerate(qna_records, start=1):
        question = record.get("question", "Untitled question")
        answer = record.get("answer", "")

        with st.expander(
            f"{index}. {question}",
            expanded=(index == 1),
        ):
            st.markdown(answer or "_No answer saved._")