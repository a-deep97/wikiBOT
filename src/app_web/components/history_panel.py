
"""Conversation history panel for wikiBOT."""

import streamlit as st

from src.backend.chat_manager import ChatManager
from src.app_web.styles import render_page_header


def _get_chat_manager() -> ChatManager | None:
    """Return a chat manager for the authenticated user."""

    client = st.session_state.get("supabase_client")
    user = st.session_state.get("supabase_user")

    if not client or not user:
        return None

    return ChatManager(
        client=client,
        user_id=str(user.id),
    )


def _restore_chat(
    backend,
    chat: dict,
    manager: ChatManager,
) -> None:
    """Load a conversation and restore it in Streamlit session state."""

    try:
        article_title = chat["article_title"]

        # Load the article into the current backend.
        with st.spinner(f"Loading {article_title}..."):
            backend.load_knowledge(article_title)

        # Fetch the saved messages.
        saved_messages = manager.get_messages(chat["id"])

        restored_messages = []
        previous_question = None

        for message in saved_messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                previous_question = content

                restored_messages.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )

            elif role == "assistant":
                restored_messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "question": previous_question or "",
                        "saved": False,
                        "sources": message.get("sources") or [],
                    }
                )

                previous_question = None

        # Restore the conversation.
        st.session_state.messages = restored_messages
        st.session_state.article_title = article_title

        # Synchronize the sidebar search box with the restored article.
        st.session_state.pending_article = article_title
        st.session_state.searchbox_version = (
            st.session_state.get("searchbox_version", 0) + 1
        )

        st.session_state.active_chat_id = chat["id"]
        st.session_state.active_section = "Chat"

        st.toast("Conversation restored!", icon="💬")
        st.rerun()

    except Exception as error:
        st.error(f"Could not restore conversation: {error}")


def _delete_chat(
    chat_id: str,
    manager: ChatManager,
) -> None:
    """Delete a conversation from Supabase."""

    try:
        manager.delete_chat(chat_id)

        # If the deleted conversation was active, clear it.
        if st.session_state.get("active_chat_id") == chat_id:
            st.session_state.messages = []
            st.session_state.active_chat_id = None

        st.session_state.pending_delete_chat_id = None

        st.toast("Conversation deleted.", icon="🗑️")
        st.rerun()

    except Exception as error:
        st.error(f"Could not delete conversation: {error}")


def render_history_panel(backend) -> None:
    """Render the authenticated user's conversation history."""

    render_page_header(
        title="Conversation History",
        subtitle="Browse and restore your previous conversations.",
        eyebrow="YOUR RESEARCH",
    )

    manager = _get_chat_manager()

    # Authentication is optional for the rest of wikiBOT.
    if manager is None:
        st.info(
            "🔐 Please log in to view your conversation history. "
            "You can continue using Wikipedia search and chat without logging in."
        )
        return

    try:
        chats = manager.list_chats()
    except Exception as error:
        st.error(f"Could not load conversation history: {error}")
        return

    if not chats:
        st.info(
            "You don't have any saved conversations yet. "
            "Start chatting with an article to build your history."
        )
        return

    st.caption(f"{len(chats)} saved conversation(s)")

    for chat in chats:
        chat_id = chat["id"]
        title = chat.get("title") or "Untitled conversation"
        article_title = chat.get("article_title") or "Unknown article"
        updated_at = chat.get("updated_at")

        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(f"📖 {article_title}")

            if updated_at:
                st.caption(f"Last updated: {updated_at}")

            open_col, delete_col = st.columns([3, 1])

            with open_col:
                if st.button(
                    "Open conversation",
                    key=f"open_chat_{chat_id}",
                    use_container_width=True,
                    type="primary",
                ):
                    _restore_chat(
                        backend=backend,
                        chat=chat,
                        manager=manager,
                    )

            with delete_col:
                if st.button(
                    "🗑️",
                    key=f"delete_chat_{chat_id}",
                    help="Delete this conversation",
                    use_container_width=True,
                ):
                    st.session_state.pending_delete_chat_id = chat_id

            if (
                st.session_state.get("pending_delete_chat_id")
                == chat_id
            ):
                st.warning(
                    "Are you sure you want to delete this conversation? "
                    "This action cannot be undone."
                )

                confirm_col, cancel_col = st.columns(2)

                with confirm_col:
                    if st.button(
                        "Confirm delete",
                        key=f"confirm_delete_{chat_id}",
                        type="primary",
                        use_container_width=True,
                    ):
                        _delete_chat(chat_id, manager)

                with cancel_col:
                    if st.button(
                        "Cancel",
                        key=f"cancel_delete_{chat_id}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_delete_chat_id = None
                        st.rerun()
