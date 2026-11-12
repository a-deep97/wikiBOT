"""Authentication services for wikiBOT."""

from supabase import Client

from src.database.postgres_client import create_supabase_client


def sign_in(email: str, password: str) -> tuple[Client, object]:
    """
    Authenticate a user with Supabase.

    Returns:
        A user-specific Supabase client and the authenticated user.
    """
    client = create_supabase_client()

    response = client.auth.sign_in_with_password(
        {
            "email": email.strip(),
            "password": password,
        }
    )

    user = response.user

    if user is None or response.session is None:
        raise ValueError("Authentication failed. Please check your credentials.")

    return client, user


def sign_out(client: Client) -> None:
    """Sign out the current user."""
    client.auth.sign_out()