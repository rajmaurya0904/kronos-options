"""
Shared utilities: IST timezone helpers, lot sizes, holiday calendar, logging setup.
"""
from __future__ import annotations
import logging
import logging.handlers
import os
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Optional

import pytz
import yaml

IST = pytz.timezone("Asia/Kolkata")

# ── NSE/BSE Holiday Calendar (update annually) ────────────────────────
# Source: NSE official holiday list.
NSE_HOLIDAYS: set[date] = {
    # 2024
    date(2024, 1, 22), date(2024, 3, 25), date(2024, 3, 29),
    date(2024, 4, 14), date(2024, 4, 17), date(2024, 5, 23),
    date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15),
    date(2024, 10, 2), date(2024, 10, 24), date(2024, 11, 15),
    date(2024, 12, 25),
    # 2025
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 4, 14),
    date(2025, 4, 17), date(2025, 8, 15), date(2025, 10, 2),
    # 2026 — source: NSE holiday calendar
    date(2026, 1, 1),  date(2026, 1, 26), date(2026, 3, 3),
    date(2026, 4, 2),  date(2026, 4, 14), date(2026, 5, 1),
    date(2026, 5, 24), date(2026, 8, 15), date(2026, 10, 2),
    date(2026, 11, 5), date(2026, 11, 14), date(2026, 12, 25),
}

# Current lot sizes (update when NSE revises).
# ASSUMPTION: These are best-known values as of June 2026.
# At runtime, prefer fetching from broker API to catch changes.
LOT_SIZES: dict[str, int] = {
    "NIFTY":    75,
    "BANKNIFTY": 15,
    "SENSEX":   10,
}

MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


# ── Time helpers ──────────────────────────────────────────────────────

def now_ist() -> datetime:
    return datetime.now(IST)


def is_trading_day(d: Optional[date] = None) -> bool:
    d = d or now_ist().date()
    return d.weekday() < 5 and d not in NSE_HOLIDAYS


def is_market_open(dt: Optional[datetime] = None) -> bool:
    dt = dt or now_ist()
    t = dt.time()
    return is_trading_day(dt.date()) and MARKET_OPEN <= t < MARKET_CLOSE


def ist_to_str(dt: datetime, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if dt.tzinfo is None:
        dt = IST.localize(dt)
    return dt.astimezone(IST).strftime(fmt)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ── Strike rounding ───────────────────────────────────────────────────

def round_to_atm(price: float, atm_step: int) -> int:
    return int(round(price / atm_step) * atm_step)


# ── Config loader ─────────────────────────────────────────────────────

_CONFIG_CACHE: Optional[dict] = None

def load_config(path: str = "config.yaml") -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    cfg_path = Path(path)
    if not cfg_path.exists():
        cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


# ── Instrument key helper ─────────────────────────────────────────────

def get_instrument_key(symbol: str, config_path: str = "config.yaml") -> str:
    """
    Return the correct instrument key for the configured broker.

    - Upstox  → instruments.<symbol>.upstox_key
    - Zerodha → instruments.<symbol>.zerodha_symbol

    Falls back to upstox_key when zerodha_symbol is missing.
    """
    cfg = load_config(config_path)
    broker = cfg.get("broker", {}).get("primary", "upstox")
    inst = cfg["instruments"][symbol]
    if broker == "zerodha":
        return inst.get("zerodha_symbol") or inst["upstox_key"]
    return inst["upstox_key"]


# ── Logging setup ─────────────────────────────────────────────────────

def setup_logging(name: str = "kronos_options", config_path: str = "config.yaml") -> logging.Logger:
    cfg = load_config(config_path).get("logging", {})
    level = getattr(logging, cfg.get("level", "INFO"))
    log_dir = Path(cfg.get("log_dir", "logs/"))
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if cfg.get("log_to_file", True):
        fh = logging.handlers.RotatingFileHandler(
            log_dir / f"{name}.log",
            maxBytes=cfg.get("max_bytes", 10_485_760),
            backupCount=cfg.get("backup_count", 5),
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Broker factory ────────────────────────────────────────────────────

def get_broker(config_path: str = "config.yaml"):
    """Instantiate the configured broker from .env credentials."""
    from dotenv import load_dotenv
    load_dotenv()

    cfg = load_config(config_path)
    primary = cfg.get("broker", {}).get("primary", "upstox")

    if primary == "upstox":
        from src.broker.upstox import UpstoxBroker
        token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
        if not token:
            raise EnvironmentError(
                "UPSTOX_ACCESS_TOKEN not set in .env\n"
                "Get it from: https://login.upstox.com → API → Access Token"
            )
        return UpstoxBroker(
            access_token=token,
            request_delay_s=cfg["broker"].get("request_delay_s", 0.35),
            timeout_s=cfg["broker"].get("timeout_s", 10),
            max_retries=cfg["broker"].get("max_retries", 3),
        )

    elif primary == "zerodha":
        from src.broker.zerodha import ZerodhaBroker
        api_key = os.environ.get("ZERODHA_API_KEY", "")
        access_token = os.environ.get("ZERODHA_ACCESS_TOKEN", "")
        if not api_key:
            raise EnvironmentError(
                "ZERODHA_API_KEY not set in .env\n"
                "Get it from: https://kite.trade → My Apps → API Key"
            )
        if not access_token:
            raise EnvironmentError(
                "ZERODHA_ACCESS_TOKEN not set in .env\n"
                "Generate it each morning via Kite Connect login flow."
            )
        return ZerodhaBroker(api_key=api_key, access_token=access_token)

    else:
        raise ValueError(
            f"Unknown broker '{primary}' in config.yaml. "
            "Valid values: 'upstox', 'zerodha'"
        )
