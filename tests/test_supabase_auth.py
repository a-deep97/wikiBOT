
"""Test Supabase authentication for wikiBOT."""

from getpass import getpass

from src.services.auth_service import sign_in, sign_out


def main():
    email = input("Enter your invited email: ").strip()
    password = getpass("Enter your password: ")

    try:
        # Authenticate the user.
        client, user = sign_in(email, password)

        print("\nAuthentication successful!")
        print(f"User ID: {user.id}")
        print(f"Email: {user.email}")

        # Test access to the chats table.
        response = (
            client.table("chats")
            .select("id")
            .limit(5)
            .execute()
        )

        print("\nSupabase database access successful!")
        print(f"Visible chats: {len(response.data)}")

        sign_out(client)
        print("\nUser signed out successfully.")

    except Exception as error:
        print(f"\nAuthentication test failed: {error}")


if __name__ == "__main__":
    main()
