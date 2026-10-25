"""Shared styling for the wikiBOT Streamlit application."""

import streamlit as st


def apply_global_styles() -> None:
    """Apply the shared wikiBOT theme."""

    st.markdown(
        """
        <style>
            :root {
                --wikibot-accent: #4F7DF3;
                --wikibot-accent-hover: #3B66D9;
                --wikibot-border: rgba(128, 128, 128, 0.25);
                --wikibot-muted: #808897;
            }

            .block-container {
                max-width: 1440px;
                padding-top: 1.8rem;
                padding-bottom: 3rem;
            }

            h1, h2, h3 {
                letter-spacing: -0.025em;
            }

            .wikibot-eyebrow {
                color: var(--wikibot-accent);
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }

            .wikibot-muted {
                color: var(--wikibot-muted);
                font-size: 0.9rem;
            }

            .wikibot-card {
                border: 1px solid var(--wikibot-border);
                border-radius: 14px;
                padding: 1rem 1.15rem;
                margin-bottom: 0.8rem;
            }

            div[data-testid="stSidebar"] {
                border-right: 1px solid var(--wikibot-border);
            }

            div[data-testid="stButton"] > button {
                border-radius: 10px;
                transition: border-color 0.15s ease;
            }

            div[data-testid="stTextInput"] input {
                border-radius: 10px;
            }

            div[data-testid="stChatMessage"] {
                border-radius: 12px;
            }

            .wikibot-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.4rem 0 1rem; }
            .wikibot-chip {
                border: 1px solid var(--wikibot-border);
                border-radius: 999px;
                padding: 0.15rem 0.7rem;
                font-size: 0.8rem;
                color: var(--wikibot-muted);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str,
    eyebrow: str = "WIKIBOT KNOWLEDGE BASE",
) -> None:
    """Render a consistent page heading."""

    st.markdown(
        f"""
        <div class="wikibot-eyebrow">{eyebrow}</div>
        """,
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f'<div class="wikibot-muted">{subtitle}</div>',
        unsafe_allow_html=True,
    )