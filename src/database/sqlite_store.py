import sqlite3
from pathlib import Path


class SQLiteStore:
    """
    Persistent local storage for Wikipedia articles,
    chunks, links, and structured article sections.
    """

    def __init__(self, db_path: str = "data/wikibot.db"):

        self.db_path = Path(db_path)

        # Create the parent directory if it doesn't exist.
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False
        )

        # Enable foreign-key enforcement.
        self.connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._create_tables()

    def _create_tables(self):

        cursor = self.connection.cursor()

        # --------------------------------------------------
        # Articles
        # --------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE NOT NULL,
                fetched_at TEXT NOT NULL
            )
        """)

        # --------------------------------------------------
        # RAG chunks
        # --------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                section TEXT,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,

                FOREIGN KEY(article_id)
                    REFERENCES articles(id)
                    ON DELETE CASCADE,

                UNIQUE(article_id, chunk_index)
            )
        """)

        # --------------------------------------------------
        # Article links
        # --------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_article_id INTEGER NOT NULL,
                target_title TEXT NOT NULL,

                FOREIGN KEY(source_article_id)
                    REFERENCES articles(id)
                    ON DELETE CASCADE,

                UNIQUE(source_article_id, target_title)
            )
        """)

        # --------------------------------------------------
        # Structured article sections
        # --------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS article_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_title TEXT NOT NULL,
                section_order INTEGER NOT NULL,
                section_title TEXT NOT NULL,
                section_text TEXT NOT NULL,

                UNIQUE(article_title, section_order)
            )
        """)

        self.connection.commit()

    # ------------------------------------------------------
    # Articles
    # ------------------------------------------------------

    def get_article(self, title: str):

        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT id, title, fetched_at
            FROM articles
            WHERE LOWER(title) = LOWER(?)
        """, (title,))

        return cursor.fetchone()

    def save_article(
        self,
        title: str,
        fetched_at: str
    ) -> int:

        cursor = self.connection.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO articles (
                title,
                fetched_at
            )
            VALUES (?, ?)
        """, (
            title,
            fetched_at
        ))

        self.connection.commit()

        article = self.get_article(title)

        return article[0]

    # ------------------------------------------------------
    # Chunks
    # ------------------------------------------------------

    def save_chunks(
        self,
        article_id: int,
        chunks: list[dict]
    ):

        cursor = self.connection.cursor()

        for chunk in chunks:

            cursor.execute("""
                INSERT OR IGNORE INTO chunks (
                    article_id,
                    section,
                    chunk_index,
                    content
                )
                VALUES (?, ?, ?, ?)
            """, (
                article_id,
                chunk["section"],
                chunk["chunk_index"],
                chunk["content"]
            ))

        self.connection.commit()

    def get_chunks(
        self,
        article_id: int
    ) -> list[dict]:

        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT
                section,
                chunk_index,
                content
            FROM chunks
            WHERE article_id = ?
            ORDER BY id
        """, (article_id,))

        rows = cursor.fetchall()

        return [
            {
                "section": row[0],
                "chunk_index": row[1],
                "content": row[2]
            }
            for row in rows
        ]

    # ------------------------------------------------------
    # Links
    # ------------------------------------------------------

    def save_links(
        self,
        article_id: int,
        links: list[str]
    ):

        cursor = self.connection.cursor()

        for link in links:

            cursor.execute("""
                INSERT OR IGNORE INTO links (
                    source_article_id,
                    target_title
                )
                VALUES (?, ?)
            """, (
                article_id,
                link
            ))

        self.connection.commit()

    def get_links(
        self,
        article_id: int
    ) -> list[str]:

        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT target_title
            FROM links
            WHERE source_article_id = ?
        """, (article_id,))

        return [
            row[0]
            for row in cursor.fetchall()
        ]

    # ------------------------------------------------------
    # Article sections
    # ------------------------------------------------------

    def save_article_sections(
        self,
        article_title: str,
        sections: list[dict[str, str]],
    ) -> None:
        """
        Save the structured sections of an article.

        Existing sections for the article are replaced so
        the cache always represents the latest loaded article.
        """

        cursor = self.connection.cursor()

        # Remove existing cached sections.
        cursor.execute(
            """
            DELETE FROM article_sections
            WHERE article_title = ?
            """,
            (article_title,),
        )

        # Save sections in their original order.
        for index, section in enumerate(sections):

            cursor.execute(
                """
                INSERT INTO article_sections (
                    article_title,
                    section_order,
                    section_title,
                    section_text
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    article_title,
                    index,
                    section["title"],
                    section["text"],
                ),
            )

        self.connection.commit()

    def get_article_sections(
        self,
        article_title: str,
    ) -> list[dict[str, str]]:
        """
        Return cached sections for an article
        in their original order.
        """

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT
                section_title,
                section_text
            FROM article_sections
            WHERE article_title = ?
            ORDER BY section_order
            """,
            (article_title,),
        )

        rows = cursor.fetchall()

        return [
            {
                "title": row[0],
                "text": row[1],
            }
            for row in rows
        ]

    def has_article_sections(
        self,
        article_title: str
    ) -> bool:
        """Return whether cached sections exist for an article."""

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT 1
            FROM article_sections
            WHERE article_title = ?
            LIMIT 1
            """,
            (article_title,),
        )

        return cursor.fetchone() is not None

    # ------------------------------------------------------
    # Utility
    # ------------------------------------------------------

    def article_exists(
        self,
        title: str
    ) -> bool:

        return self.get_article(title) is not None

    def get_stats(self) -> dict:

        cursor = self.connection.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM articles"
        )
        articles = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM chunks"
        )
        chunks = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM links"
        )
        links = cursor.fetchone()[0]

        return {
            "articles": articles,
            "chunks": chunks,
            "links": links
        }

    def close(self):
        self.connection.close()