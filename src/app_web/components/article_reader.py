"""Article reader UI for wikiBOT."""

import streamlit as st

from src.app_web.helpers import render_article_sections
from src.app_web.styles import render_page_header


def render_article_reader(
    backend,
    article_title: str | None,
) -> None:
    """Render the currently loaded Wikipedia article."""

    render_page_header(
        title=article_title or "Article Reader",
        subtitle=(
            "Explore the article, review its sections, "
            "and use the same knowledge base to ask questions."
        ),
    )

    if not backend or not article_title:
        st.info(
            "Search for a Wikipedia article in the sidebar "
            "to start exploring its content."
        )

        st.markdown(
            """
            <div class="wikibot-card">
                <h3>📖 Your research workspace</h3>
                <p>
                    Load an article to read its sections and use
                    wikiBOT to ask questions about the content.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Keep the article display separate from the page heading.
    with st.container(border=True):
        st.markdown("#### 📄 Article content")
        st.caption(
            "Browse the sections below. Content comes from "
            "your existing article store."
        )
        st.divider()

        try:
            render_article_sections(backend)
        except Exception as error:
            st.error(f"Unable to render article sections: {error}")