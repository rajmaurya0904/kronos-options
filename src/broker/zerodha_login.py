"""
zerodha_login.py — One-shot daily login to get a fresh access token.

Run every morning before starting the paper/live trader:
    python -m src.broker.zerodha_login

This will:
  1. Open the Kite login URL in your browser.
  2. Ask you to paste the redirect URL (contains request_token).
  3. Exchange for access_token and write it to .env automatically.

Requires:
    ZERODHA_API_KEY and ZERODHA_API_SECRET in .env
"""
from __future__ import annotations
import os
import re
import webbrowser
from pathlib import Path


def main():
    from dotenv import load_dotenv, set_key
    load_dotenv()

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("Install kiteconnect: pip install kiteconnect")
        return

    api_key    = os.environ.get("ZERODHA_API_KEY", "")
    api_secret = os.environ.get("ZERODHA_API_SECRET", "")

    if not api_key or not api_secret:
        print("Set ZERODHA_API_KEY and ZERODHA_API_SECRET in .env first.")
        return

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    print(f"\nOpening Kite login URL:\n{login_url}\n")
    webbrowser.open(login_url)

    redirect_url = input(
        "After login, paste the full redirect URL here:\n> "
    ).strip()

    # Extract request_token from URL
    match = re.search(r"request_token=([^&]+)", redirect_url)
    if not match:
        print("Could not extract request_token from URL. Make sure you pasted the full URL.")
        return

    request_token = match.group(1)
    print(f"Request token: {request_token}")

    session = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session["access_token"]
    print(f"\nAccess token generated: {access_token[:8]}...")

    # Write to .env
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text("")

    set_key(str(env_path), "ZERODHA_ACCESS_TOKEN", access_token)
    print(f"\n✅ ZERODHA_ACCESS_TOKEN written to .env")
    print("You can now start the trader:")
    print("    python -m src.paper_trader")


if __name__ == "__main__":
    main()
