
"""Supabase persistence for user-specific Q&A records."""

from supabase import Client


class SupabaseQnAManager:
    """Manage Q&A records belonging to an authenticated user."""

    def __init__(self, client: Client, user_id: str):
        self.client = client
        self.user_id = user_id

    def add(
        self,
        article_title: str,
        question: str,
        answer: str,
    ) -> bool:
        """Save a Q&A pair, avoiding duplicate records."""

        article_title = (article_title or "").strip()
        question = (question or "").strip()
        answer = (answer or "").strip()

        if not (article_title and question and answer):
            raise ValueError(
                "Article title, question, and answer are required."
            )

        existing = (
            self.client.table("qna")
            .select("id")
            .eq("user_id", self.user_id)
            .ilike("article_title", article_title)
            .eq("question", question)
            .eq("answer", answer)
            .limit(1)
            .execute()
        )

        if existing.data:
            return False

        self.client.table("qna").insert({
            "user_id": self.user_id,
            "article_title": article_title,
            "question": question,
            "answer": answer,
        }).execute()

        return True

    def get(
        self,
        article_title: str | None = None,
    ) -> list[dict]:
        """Return this user's Q&A records, newest first."""

        query = (
            self.client.table("qna")
            .select("id, article_title, question, answer, created_at")
            .eq("user_id", self.user_id)
            .order("created_at", desc=True)
        )

        if article_title is not None:
            query = query.ilike("article_title", article_title.strip())

        response = query.execute()
        return response.data or []

    def delete(self, qna_id: str) -> bool:
        """Delete a Q&A record belonging to this user."""

        response = (
            self.client.table("qna")
            .delete()
            .eq("id", qna_id)
            .eq("user_id", self.user_id)
            .execute()
        )

        return bool(response.data)
