"""
zerodha_auth.py — Daily Zerodha access token refresh.

Zerodha Kite Connect access tokens expire at 6:00 AM IST every day.
Run this script once each morning before starting the bot.

Two modes:
  1. Automated (TOTP-based): requires ZERODHA_TOTP_SECRET in .env
     python zerodha_auth.py

  2. Manual (browser-based): opens login URL in browser, paste request_token
     python zerodha_auth.py --manual

On success:
  - Writes ZERODHA_ACCESS_TOKEN to .env (updates in-place)
  - Saves today's NFO instruments snapshot to data/zerodha_instruments/
    (used by the backtester for expired option token resolution)
  - Prints confirmation with token expiry time

Prerequisites:
  pip install kiteconnect pyotp python-dotenv requests
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import webbrowser
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

ENV_FILE = Path(".env")
IST = timezone(timedelta(hours=5, minutes=30))


# ── .env helpers ──────────────────────────────────────────────────────

def _load_env() -> dict[str, str]:
    """Load .env file into a dict. Does not use dotenv to avoid side effects."""
    if not ENV_FILE.exists():
        return {}
    env = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip()
    return env


def _update_env_token(new_token: str) -> None:
    """
    Update ZERODHA_ACCESS_TOKEN in .env in-place.
    Creates .env from .env.example if it does not exist.
    """
    if not ENV_FILE.exists():
        example = Path(".env.example")
        if example.exists():
            ENV_FILE.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
            print(".env created from .env.example")
        else:
            ENV_FILE.write_text("ZERODHA_ACCESS_TOKEN=\n", encoding="utf-8")

    content = ENV_FILE.read_text(encoding="utf-8")
    pattern = r"^ZERODHA_ACCESS_TOKEN=.*$"
    replacement = f"ZERODHA_ACCESS_TOKEN={new_token}"

    if re.search(pattern, content, flags=re.MULTILINE):
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    else:
        content = content.rstrip("\n") + f"\n{replacement}\n"

    ENV_FILE.write_text(content, encoding="utf-8")
    print(f"✅ ZERODHA_ACCESS_TOKEN updated in {ENV_FILE}")


# ── NFO instrument snapshot ───────────────────────────────────────────

def _save_nfo_snapshot(api_key: str, access_token: str) -> None:
    """
    Save today's NFO instruments snapshot immediately after token refresh.
    This snapshot is used by the backtester to resolve expired option tokens.
    Skips silently if kiteconnect is unavailable or snapshot already exists.
    """
    try:
        from src.broker.zerodha_historical import save_snapshot
        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)

        today = datetime.now(IST).strftime("%Y-%m-%d")
        print(f"\n📸 Saving NFO instrument snapshot for {today}...")
        path = save_snapshot(kite, today)
        print(f"   Snapshot saved → {path}")

    except ImportError as e:
        logger.warning("Snapshot skipped (import error): %s", e)
        print(f"   ⚠️  Snapshot skipped: {e}")
    except Exception as e:
        # Non-fatal: token refresh succeeded, snapshot is best-effort
        logger.warning("Snapshot save failed (non-fatal): %s", e)
        print(f"   ⚠️  Snapshot save failed (non-fatal): {e}")
        print("      Run manually: python -m src.broker.zerodha_historical --save-snapshot")


# ── TOTP-based automated login ────────────────────────────────────────

def _get_totp_code(totp_secret: str) -> str:
    """Generate current TOTP code using the secret."""
    try:
        import pyotp
    except ImportError:
        raise ImportError("Run: pip install pyotp")
    return pyotp.TOTP(totp_secret).now()


def _automated_login(api_key: str, api_secret: str, user_id: str, password: str, totp_secret: str) -> str:
    """
    Perform headless login using requests + TOTP.
    Returns request_token extracted from the redirect URL.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    # Step 1: POST credentials to Kite login
    login_url = "https://kite.zerodha.com/api/login"
    resp = session.post(login_url, data={"user_id": user_id, "password": password}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Login step 1 failed: {data}")
    request_id = data["data"]["request_id"]
    logger.info("Login step 1 passed. request_id=%s", request_id)

    # Step 2: POST TOTP to complete 2FA
    totp_url = "https://kite.zerodha.com/api/twofa"
    totp_code = _get_totp_code(totp_secret)
    resp = session.post(
        totp_url,
        data={"user_id": user_id, "request_id": request_id, "twofa_value": totp_code, "twofa_type": "totp"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"2FA step failed: {data}")
    logger.info("2FA passed.")

    # Step 3: Follow redirect to get request_token from URL
    login_redirect = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
    resp = session.get(login_redirect, allow_redirects=True, timeout=15)
    final_url = resp.url
    match = re.search(r"request_token=([^&]+)", final_url)
    if not match:
        raise RuntimeError(
            f"Could not extract request_token from redirect URL: {final_url}\n"
            "This may happen if your Kite app redirect URI is not set to 'https://127.0.0.1/callback'."
        )
    request_token = match.group(1)
    logger.info("request_token obtained.")
    return request_token


# ── Manual browser-based login ────────────────────────────────────────

def _manual_login(api_key: str) -> str:
    """
    Open Kite login URL in browser. User logs in and pastes back the
    request_token from the redirect URL.
    """
    login_url = f"https://kite.zerodha.com/connect/login?api_key={api_key}&v=3"
    print(f"\nOpening Zerodha login in browser:\n{login_url}")
    webbrowser.open(login_url)
    print("\nAfter logging in, you will be redirected to a URL like:")
    print("  https://127.0.0.1/callback?request_token=XXXXX&action=login&status=success")
    print("\nCopy the full redirect URL and paste it below.")
    redirect_url = input("Paste redirect URL: ").strip()
    match = re.search(r"request_token=([^&]+)", redirect_url)
    if not match:
        raise ValueError(f"Could not find request_token in: {redirect_url}")
    return match.group(1)


# ── Token exchange ─────────────────────────────────────────────────────

def _exchange_for_access_token(api_key: str, api_secret: str, request_token: str) -> str:
    """Exchange request_token for access_token using Kite Connect SDK."""
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        raise ImportError("Run: pip install kiteconnect")

    kite = KiteConnect(api_key=api_key)
    session_data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session_data["access_token"]
    user_id      = session_data.get("user_id", "unknown")
    login_time   = session_data.get("login_time", "")
    logger.info("Access token generated for user_id=%s login_time=%s", user_id, login_time)
    return access_token


# ── Main ──────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Zerodha daily access token refresh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python zerodha_auth.py            # automated TOTP login\n"
            "  python zerodha_auth.py --manual   # browser-based login\n"
            "  python zerodha_auth.py --no-snapshot  # skip NFO snapshot\n"
        ),
    )
    parser.add_argument("--manual", action="store_true", help="Use browser-based manual login")
    parser.add_argument(
        "--no-snapshot", action="store_true",
        help="Skip saving NFO instrument snapshot (not recommended)"
    )
    args = parser.parse_args()

    # Load credentials from .env
    env = _load_env()
    api_key    = env.get("ZERODHA_API_KEY", "").strip()
    api_secret = env.get("ZERODHA_API_SECRET", "").strip()

    if not api_key or not api_secret:
        print("❌ ZERODHA_API_KEY and ZERODHA_API_SECRET must be set in .env")
        sys.exit(1)

    print(f"\n⏰ Zerodha token refresh — {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
    print(f"   api_key: {api_key[:6]}...")

    try:
        if args.manual:
            request_token = _manual_login(api_key)
        else:
            user_id     = env.get("ZERODHA_USER_ID", "").strip()
            password    = env.get("ZERODHA_PASSWORD", "").strip()
            totp_secret = env.get("ZERODHA_TOTP_SECRET", "").strip()

            missing = [k for k, v in {
                "ZERODHA_USER_ID":     user_id,
                "ZERODHA_PASSWORD":    password,
                "ZERODHA_TOTP_SECRET": totp_secret,
            }.items() if not v]

            if missing:
                print(f"❌ Missing for automated login: {', '.join(missing)}")
                print("   Either add them to .env OR run with --manual flag.")
                sys.exit(1)

            print("   Mode: automated (TOTP)")
            request_token = _automated_login(api_key, api_secret, user_id, password, totp_secret)

        # Exchange request_token for access_token
        print("\n🔄 Exchanging request_token for access_token...")
        access_token = _exchange_for_access_token(api_key, api_secret, request_token)

        # Write token to .env
        _update_env_token(access_token)

        # Save NFO snapshot (unless --no-snapshot flag set)
        if not args.no_snapshot:
            _save_nfo_snapshot(api_key, access_token)
        else:
            print("   ⏭️  Snapshot skipped (--no-snapshot flag set)")

        # Show expiry info (tokens expire at 06:00 IST next day)
        now_ist = datetime.now(IST)
        expiry_ist = now_ist.replace(hour=6, minute=0, second=0, microsecond=0)
        if now_ist.hour >= 6:
            expiry_ist = expiry_ist + timedelta(days=1)
        print(f"\n   Token valid until: {expiry_ist.strftime('%Y-%m-%d 06:00 IST')}")
        print("\n✅ Done. You can now start the bot.")

    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)
    except Exception as e:
        logger.error("Token refresh failed: %s", e)
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
