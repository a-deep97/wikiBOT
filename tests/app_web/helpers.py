
import streamlit as st


def render_article_sections(backend):
    """Render the currently loaded article sections."""

    sections = backend.get_article_sections()

    if not sections:
        return

    st.subheader("Article")

    for section in sections:
        section_title = section["title"].strip()
        section_text = section["text"].strip()

        if not section_title:
            continue

        with st.expander(
            section_title,
            expanded=False,
        ):
            if section_text:
                st.write(section_text)
            else:
                st.caption("No content available.")