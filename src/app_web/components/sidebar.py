
"""Sidebar navigation, article search, and optional authentication."""

import streamlit as st

from src.app_web.components.search_panel import render_search_panel
from src.services.auth_service import sign_in, sign_out


def render_auth_section() -> None:
    """Render optional login/logout controls."""

    st.markdown("### 👤 Account")

    user = st.session_state.get("supabase_user")

    if user:
        email = getattr(user, "email", None) or "Authenticated user"

        st.success(f"Logged in as {email}")

        if st.button(
            "Logout",
            key="logout_button",
            use_container_width=True,
        ):
            try:
                client = st.session_state.get("supabase_client")

                if client:
                    sign_out(client)

                st.session_state.pop("supabase_client", None)
                st.session_state.pop("supabase_user", None)

                # Do not carry one user's in-memory conversation
                # into another authentication state.
                st.session_state.messages = []
                st.session_state.active_chat_id = None

                st.rerun()

            except Exception as error:
                st.error(f"Could not log out: {error}")

    else:
        with st.expander("Login", expanded=False):
            with st.form("login_form"):
                email = st.text_input(
                    "Email",
                    placeholder="Enter your email",
                )

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                )

                submitted = st.form_submit_button(
                    "Login",
                    use_container_width=True,
                    type="primary",
                )

            if submitted:
                if not email.strip() or not password:
                    st.error("Please enter both email and password.")
                    return

                try:
                    client, authenticated_user = sign_in(
                        email=email,
                        password=password,
                    )

                    st.session_state.supabase_client = client
                    st.session_state.supabase_user = authenticated_user

                    # Anonymous conversations must not become
                    # part of the authenticated user's history.
                    st.session_state.messages = []
                    st.session_state.active_chat_id = None

                    st.rerun()

                except Exception:
                    st.error(
                        "Login failed. Please check your credentials "
                        "or invitation status."
                    )



def render_sidebar(
    search_function,
    recent_searches: list[str],
) -> str | None:
    """Render the sidebar and return the selected article title."""

    with st.sidebar:
        # Account section at the very top.

        st.divider()

        st.markdown("## 🧠 wikiBOT")
        st.caption("Your Wikipedia research assistant")
        st.divider()

        selected_article = render_search_panel(
            search_function=search_function,
            recent_searches=recent_searches,
        )

    return selected_article
