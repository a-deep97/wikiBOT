import sqlite3
from pathlib import Path


class SQLiteStore:
    """
    Persistent local storage for Wikipedia articles,
    chunks, and links.
    """

    def __init__(self, db_path: str = "data/wikibot.db"):

        self.db_path = Path(db_path)

        # Create the parent directory if it doesn't exist.
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.connection = sqlite3.connect(
            self.db_path
        )

        # Enable foreign-key enforcement.
        self.connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._create_tables()

    def _create_tables(self):

        cursor = self.connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT UNIQUE NOT NULL,
                fetched_at TEXT NOT NULL
            )
        """)

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

        self.connection.commit()

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