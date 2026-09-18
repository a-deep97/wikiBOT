import sqlite3


DB_PATH = "data/wikibot.db"


connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()


print("\n=== ARTICLES ===")

cursor.execute("""
    SELECT id, title, fetched_at
    FROM articles
""")

for row in cursor.fetchall():
    print(row)


print("\n=== CHUNKS ===")

cursor.execute("""
    SELECT
        id,
        article_id,
        section,
        chunk_index,
        content
    FROM chunks
    LIMIT 10
""")

for row in cursor.fetchall():
    print(
        f"\nID: {row[0]}"
        f"\nArticle ID: {row[1]}"
        f"\nSection: {row[2]}"
        f"\nChunk: {row[3]}"
        f"\nContent: {row[4][:150]}..."
    )


print("\n=== LINKS ===")

cursor.execute("""
    SELECT
        id,
        source_article_id,
        target_title
    FROM links
    LIMIT 20
""")

for row in cursor.fetchall():
    print(row)


connection.close()