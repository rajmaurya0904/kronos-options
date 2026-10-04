"""
Shared utilities: IST timezone helpers, holiday calendar, event calendar, logging setup.
Lot sizes live in config.yaml (and, in the backtester, come from each contract).
"""
from __future__ import annotations
import logging
import logging.handlers
import math
import os
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Optional

import pytz
import yaml

IST = pytz.timezone("Asia/Kolkata")

# ── NSE/BSE Holiday Calendar (update annually) ────────────────────────
# Weekday trading holidays only (equity + equity derivatives).
# Source: NSE official holiday circulars. Update every December.
NSE_HOLIDAYS: set[date] = {
    # 2024
    date(2024, 1, 22), date(2024, 1, 26), date(2024, 3, 8),
    date(2024, 3, 25), date(2024, 3, 29), date(2024, 4, 11),
    date(2024, 4, 17), date(2024, 5, 1),  date(2024, 5, 20),
    date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15),
    date(2024, 10, 2), date(2024, 11, 1), date(2024, 11, 15),
    date(2024, 11, 20), date(2024, 12, 25),
    # 2025
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18),
    date(2025, 5, 1),  date(2025, 8, 15), date(2025, 8, 27),
    date(2025, 10, 2), date(2025, 10, 21), date(2025, 10, 22),
    date(2025, 11, 5), date(2025, 12, 25),
    # 2026
    date(2026, 1, 26), date(2026, 3, 3),  date(2026, 3, 26),
    date(2026, 3, 31), date(2026, 4, 3),  date(2026, 4, 14),
    date(2026, 5, 1),  date(2026, 5, 28), date(2026, 6, 26),
    date(2026, 9, 14), date(2026, 10, 2), date(2026, 10, 20),
    date(2026, 11, 10), date(2026, 11, 24), date(2026, 12, 25),
}

MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


# ── Time helpers ──────────────────────────────────────────────────────

def now_ist() -> datetime:
    return datetime.now(IST)


def today_ist() -> date:
    """Today's date in India, whatever timezone the machine runs in."""
    return now_ist().date()


def is_event_day(d: date, config: dict) -> bool:
    """True if d is in config.yaml event_calendar (RBI policy etc.)."""
    events = config.get("event_calendar") or {}
    return d.strftime("%Y-%m-%d") in {str(k) for k in events}


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
    # Half-up, not Python's round(): banker's rounding sends 24325 to 24300
    # but 24375 to 24400, so the ATM strike flipped direction at midpoints.
    return int(math.floor(price / atm_step + 0.5) * atm_step)


# ── Config loader ─────────────────────────────────────────────────────

_CONFIG_CACHE: Optional[dict] = None

def load_config(path: str = "config.yaml") -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    cfg_path = Path(path)
    if not cfg_path.exists():
        # Try relative to this file's parent-parent (project root)
        cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


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

    # Handlers go on the root logger: every module logs via getLogger(__name__)
    # ("src.paper_trader", "src.backtester", ...), so a handler on a logger
    # called `name` would never see them and INFO trade logs were dropped.
    logger = logging.getLogger()
    logger.setLevel(level)
    if getattr(logger, "_kronos_configured", False):  # idempotent on re-entry
        return logger
    logger._kronos_configured = True

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
            raise EnvironmentError("UPSTOX_ACCESS_TOKEN not set in .env")
        return UpstoxBroker(
            access_token=token,
            request_delay_s=cfg["broker"].get("request_delay_s", 0.35),
            timeout_s=cfg["broker"].get("timeout_s", 10),
            max_retries=cfg["broker"].get("max_retries", 3),
        )
    elif primary == "zerodha":
        from src.broker.zerodha import ZerodhaBroker
        return ZerodhaBroker(
            api_key=os.environ["ZERODHA_API_KEY"],
            access_token=os.environ["ZERODHA_ACCESS_TOKEN"],
        )
    else:
        raise ValueError(f"Unknown broker: {primary}")
