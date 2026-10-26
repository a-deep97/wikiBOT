"""Saving and loading of question-and-answer records."""

import sqlite3
from datetime import datetime, timezone


class QnAManager:
    """
    Manage Q&A pairs that a user saved from the chat.

    Records are stored in the ``qna`` table of the wikiBOT SQLite
    database. The manager shares the connection of ``SQLiteStore``, so
    no second database file or connection is needed, and it creates its
    own table on startup if it does not exist yet.
    """

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self._create_table()

    def _create_table(self) -> None:
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS qna (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_title TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL,

                UNIQUE(article_title, question, answer)
            )
        """)

        self.connection.commit()

    def add(
        self,
        article_title: str,
        question: str,
        answer: str,
    ) -> bool:
        """
        Save a Q&A pair for an article.

        Returns:
            True if a new record was stored, False if an identical
            record had already been saved.

        Raises:
            ValueError: If the title, question, or answer is empty.
        """

        article_title = (article_title or "").strip()
        question = (question or "").strip()
        answer = (answer or "").strip()

        if not (article_title and question and answer):
            raise ValueError(
                "Article title, question, and answer are all required."
            )

        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO qna (
                article_title,
                question,
                answer,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                article_title,
                question,
                answer,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        self.connection.commit()

        return cursor.rowcount > 0

    def get(self, article_title: str | None = None) -> list[dict]:
        """
        Return saved Q&A records, newest first.

        When article_title is None, records for all articles are returned.
        The title is matched case-insensitively.
        """

        if article_title is None:
            cursor = self.connection.execute(
                """
                SELECT id, article_title, question, answer, created_at
                FROM qna
                ORDER BY id DESC
                """
            )
        else:
            cursor = self.connection.execute(
                """
                SELECT id, article_title, question, answer, created_at
                FROM qna
                WHERE LOWER(article_title) = LOWER(?)
                ORDER BY id DESC
                """,
                (article_title.strip(),),
            )

        return [
            {
                "id": row[0],
                "article_title": row[1],
                "question": row[2],
                "answer": row[3],
                "created_at": row[4],
            }
            for row in cursor.fetchall()
        ]
