
"""Supabase chat persistence for wikiBOT."""

from supabase import Client


class ChatManager:
    """Manage user-specific conversations and messages."""

    def __init__(self, client: Client, user_id: str):
        self.client = client
        self.user_id = user_id

    def create_chat(
        self,
        title: str,
        article_title: str,
        model_key: str,
    ) -> str:
        """Create a conversation and return its ID."""

        response = (
            self.client.table("chats")
            .insert({
                "user_id": self.user_id,
                "title": title[:255],
                "article_title": article_title,
                "model_key": model_key,
            })
            .execute()
        )

        if not response.data:
            raise RuntimeError("Could not create the conversation.")

        return response.data[0]["id"]

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> None:
        """Save a message to a conversation owned by the current user."""

        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")

        # Verify conversation ownership.
        response = (
            self.client.table("chats")
            .select("id")
            .eq("id", chat_id)
            .eq("user_id", self.user_id)
            .execute()
        )

        if not response.data:
            raise PermissionError(
                "Conversation not found or access denied."
            )

        # Insert the message.
        (
            self.client.table("messages")
            .insert({
                "chat_id": chat_id,
                "role": role,
                "content": content,
                "sources": sources or [],
            })
            .execute()
        )

    def list_chats(self) -> list[dict]:
        """Return the current user's conversations."""

        response = (
            self.client.table("chats")
            .select("*")
            .eq("user_id", self.user_id)
            .order("updated_at", desc=True)
            .execute()
        )

        return response.data or []

    def get_messages(self, chat_id: str) -> list[dict]:
        """Return messages belonging to a conversation owned by the user."""

        # First verify that this conversation belongs to the current user.
        response = (
            self.client.table("chats")
            .select("id")
            .eq("id", chat_id)
            .eq("user_id", self.user_id)
            .execute()
        )

        if not response.data:
            raise PermissionError(
                "Conversation not found or access denied."
            )

        # Retrieve messages only after ownership is verified.
        messages_response = (
            self.client.table("messages")
            .select("*")
            .eq("chat_id", chat_id)
            .order("created_at")
            .order("id")
            .execute()
        )

        return messages_response.data or []

    def delete_chat(self, chat_id: str) -> None:
        """Delete a conversation owned by the current user."""

        (
            self.client.table("chats")
            .delete()
            .eq("id", chat_id)
            .eq("user_id", self.user_id)
            .execute()
        )
