
"""Supabase PostgreSQL connection for wikiBOT."""

import os

from dotenv import find_dotenv, load_dotenv
from supabase import Client, create_client


# Load environment variables from the project root.
load_dotenv(find_dotenv())


def create_supabase_client() -> Client:
    """
    Create a Supabase client.

    Each authenticated Streamlit session will eventually
    maintain its own client to avoid sharing authentication
    state between users.
    """

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_PUBLISHABLE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_PUBLISHABLE_KEY "
            "in environment variables."
        )

    return create_client(
        supabase_url,
        supabase_key,
    )


def test_connection() -> bool:
    """Test connectivity and access to the chats table."""

    try:
        client = create_supabase_client()

        # An unauthenticated request may return no rows
        # because of RLS. A successful query still verifies
        # connectivity and access to the table endpoint.
        client.table("chats").select("id").limit(1).execute()

        print("Successfully connected to Supabase!")
        print("The chats table is accessible.")

        return True

    except Exception as error:
        print(f"Supabase connection failed: {error}")
        return False


if __name__ == "__main__":
    test_connection()