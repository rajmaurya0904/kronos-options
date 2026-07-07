"""
Zerodha Kite Connect v3 broker implementation.

Setup:
  1. pip install kiteconnect
  2. In .env:
       ZERODHA_API_KEY=xxxx
       ZERODHA_API_SECRET=xxxx
       ZERODHA_ACCESS_TOKEN=xxxx   # refresh daily via login flow
  3. In config.yaml: broker.primary: "zerodha"

Access token must be refreshed every day.
Use the login helper:
    python -m src.broker.zerodha_login

Notes on instrument keys:
  Zerodha uses integer instrument_tokens, not string keys.
  This broker resolves string keys ("NIFTY 50", "NIFTY26JUL24750CE") → tokens
  via instruments CSV cached daily in data/zerodha_instruments.csv.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional, List

import pandas as pd

from .base import BrokerInterface

logger = logging.getLogger(__name__)

_INSTRUMENTS_CACHE_PATH = Path("data/zerodha_instruments.csv")
_INTERVAL_MAP = {
    "1minute":  "minute",
    "5minute":  "5minute",
    "15minute": "15minute",
    "30minute": "30minute",
    "60minute": "60minute",
    "day":      "day",
}


class ZerodhaBroker(BrokerInterface):
    """
    Zerodha Kite Connect v3 implementation.
    Broker primary = "zerodha" in config.yaml.
    """

    def __init__(self, api_key: str, access_token: str):
        try:
            from kiteconnect import KiteConnect
        except ImportError:
            raise ImportError(
                "kiteconnect not installed. Run: pip install kiteconnect"
            )
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self._instruments_df: Optional[pd.DataFrame] = None
        logger.info("ZerodhaBroker initialised.")

    # ── Instruments CSV (cached daily) ─────────────────────────────────

    def _load_instruments(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load NSE/NFO instruments from Zerodha or from daily cache.
        Refresh happens once per day automatically.
        """
        if self._instruments_df is not None and not refresh:
            return self._instruments_df

        today = date.today().isoformat()
        if _INSTRUMENTS_CACHE_PATH.exists():
            mtime = datetime.fromtimestamp(_INSTRUMENTS_CACHE_PATH.stat().st_mtime).date()
            if mtime.isoformat() == today and not refresh:
                self._instruments_df = pd.read_csv(_INSTRUMENTS_CACHE_PATH)
                logger.debug("Instruments loaded from cache (%d rows).", len(self._instruments_df))
                return self._instruments_df

        logger.info("Downloading fresh instruments CSV from Zerodha...")
        _INSTRUMENTS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        instruments = self.kite.instruments("NFO") + self.kite.instruments("NSE")
        df = pd.DataFrame(instruments)
        df.to_csv(_INSTRUMENTS_CACHE_PATH, index=False)
        self._instruments_df = df
        logger.info("Instruments refreshed: %d rows saved.", len(df))
        return df

    def _resolve_token(self, tradingsymbol: str, exchange: str = "NFO") -> Optional[int]:
        """Resolve tradingsymbol string → instrument_token (int)."""
        df = self._load_instruments()
        row = df[(df["tradingsymbol"] == tradingsymbol) & (df["exchange"] == exchange)]
        if row.empty:
            logger.warning("Token not found for %s:%s", exchange, tradingsymbol)
            return None
        return int(row.iloc[0]["instrument_token"])

    def _index_token(self, zerodha_symbol: str) -> Optional[int]:
        """Resolve index symbol (e.g. 'NIFTY 50') → token via NSE exchange."""
        df = self._load_instruments()
        row = df[(df["tradingsymbol"] == zerodha_symbol) & (df["exchange"] == "NSE")]
        if not row.empty:
            return int(row.iloc[0]["instrument_token"])
        # Fallback: use kite.ltp()
        try:
            resp = self.kite.ltp([f"NSE:{zerodha_symbol}"])
            return resp[f"NSE:{zerodha_symbol}"]["instrument_token"]
        except Exception as e:
            logger.error("Could not resolve index token for %s: %s", zerodha_symbol, e)
            return None

    # ── Key format helpers ────────────────────────────────────────────

    @staticmethod
    def _build_option_symbol(
        underlying: str,    # "NIFTY" | "BANKNIFTY" | "SENSEX"
        expiry: str,        # "YYYY-MM-DD"
        strike: int,
        option_type: str,   # "CE" | "PE"
    ) -> str:
        """
        Build Zerodha NFO tradingsymbol for a weekly option.
        Format: NIFTY26JUL24750CE  (SYMBOL + YY + MON + STRIKE + CE/PE)
        Monthly: NIFTY26JUL24750CE is the same; weekly adds day for near-term.

        Zerodha weekly format (non-month-end): NIFTY2571024750CE
        i.e.  NIFTY + YY + MM + DD + STRIKE + CE/PE
        """
        d = datetime.strptime(expiry, "%Y-%m-%d")
        # Check if it's the last Thursday of the month (monthly expiry)
        import calendar
        last_thu = max(
            week[3] for week in calendar.monthcalendar(d.year, d.month)
            if week[3] != 0
        )
        if d.day == last_thu:
            # Monthly format: NIFTY26JUL24750CE
            return f"{underlying}{d.strftime('%y%b').upper()}{strike}{option_type}"
        else:
            # Weekly format: NIFTY2571024750CE (yy + single-digit M + DD)
            month_char = "123456789OND"[d.month - 1]   # Zerodha: Oct=O, Nov=N, Dec=D
            return f"{underlying}{d.strftime('%y')}{month_char}{d.strftime('%d')}{strike}{option_type}"

    # ── BrokerInterface implementation ────────────────────────────────

    def get_historical_candles(
        self,
        instrument_key: str,   # zerodha_symbol e.g. "NIFTY 50"
        interval: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        token = self._index_token(instrument_key)
        if token is None:
            raise ValueError(f"Could not resolve token for {instrument_key}")

        kite_interval = _INTERVAL_MAP.get(interval, interval)
        data = self.kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=kite_interval,
            continuous=False,
            oi=False,
        )
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        })
        if df.index.tz is None:
            import pytz
            df.index = df.index.tz_localize(pytz.timezone("Asia/Kolkata"))
        return df

    def get_expired_expiries(self, instrument_key: str) -> list[str]:
        """
        Return past + upcoming expiry dates for the underlying.
        Pulls from NFO instruments CSV and returns sorted unique expiry strings.
        instrument_key: zerodha_symbol like 'NIFTY 50' or bare 'NIFTY'.
        """
        underlying = instrument_key.split()[0]   # 'NIFTY 50' → 'NIFTY'
        df = self._load_instruments()
        rows = df[
            (df["name"] == underlying) &
            (df["exchange"] == "NFO") &
            (df["instrument_type"].isin(["CE", "PE"]))
        ]
        if rows.empty:
            return []
        expiries = pd.to_datetime(rows["expiry"]).dt.date.unique()
        return sorted(str(e) for e in expiries)

    def get_expired_option_key(
        self,
        instrument_key: str,
        expiry: str,
        strike: int,
        option_type: str,
    ) -> Optional[str]:
        """Return tradingsymbol string for the given option contract."""
        underlying = instrument_key.split()[0]
        return self._build_option_symbol(underlying, expiry, strike, option_type)

    def get_expired_option_candles(
        self,
        expired_option_key: str,   # tradingsymbol e.g. "NIFTY2571024750CE"
        interval: str,
        date_str: str,
    ) -> list:
        token = self._resolve_token(expired_option_key, exchange="NFO")
        if token is None:
            return []
        kite_interval = _INTERVAL_MAP.get(interval, interval)
        try:
            return self.kite.historical_data(
                instrument_token=token,
                from_date=date_str,
                to_date=date_str,
                interval=kite_interval,
            )
        except Exception as e:
            logger.error("get_expired_option_candles failed for %s: %s", expired_option_key, e)
            return []

    def get_option_chain(
        self,
        instrument_key: str,   # zerodha_symbol like "NIFTY 50"
        expiry: str,           # "YYYY-MM-DD"
    ) -> pd.DataFrame:
        """
        Build a synthetic option chain by fetching LTP for all strikes.
        Returns DataFrame with columns: strike, CE_ltp, PE_ltp, instrument_type.
        """
        underlying = instrument_key.split()[0]
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()

        df_inst = self._load_instruments()
        rows = df_inst[
            (df_inst["name"] == underlying) &
            (df_inst["exchange"] == "NFO") &
            (pd.to_datetime(df_inst["expiry"]).dt.date == expiry_date)
        ].copy()

        if rows.empty:
            logger.warning("No instruments found for %s expiry=%s", underlying, expiry)
            return pd.DataFrame()

        # Build LTP request: max 500 instruments per call (Kite limit)
        symbols = [f"NFO:{row['tradingsymbol']}" for _, row in rows.iterrows()]
        ltps = {}
        for i in range(0, len(symbols), 500):
            chunk = symbols[i:i + 500]
            try:
                resp = self.kite.ltp(chunk)
                ltps.update(resp)
            except Exception as e:
                logger.warning("LTP batch failed: %s", e)

        records = []
        for _, row in rows.iterrows():
            sym_key = f"NFO:{row['tradingsymbol']}"
            ltp_val = ltps.get(sym_key, {}).get("last_price", 0.0)
            records.append({
                "strike":          int(row["strike"]),
                "instrument_type": row["instrument_type"],   # "CE" or "PE"
                "tradingsymbol":   row["tradingsymbol"],
                "ltp":             float(ltp_val),
            })

        chain_df = pd.DataFrame(records)
        if chain_df.empty:
            return chain_df

        # Pivot to wide format: strike, CE_ltp, PE_ltp
        ce = chain_df[chain_df["instrument_type"] == "CE"][["strike", "ltp"]].rename(columns={"ltp": "CE_ltp"})
        pe = chain_df[chain_df["instrument_type"] == "PE"][["strike", "ltp"]].rename(columns={"ltp": "PE_ltp"})
        wide = pd.merge(ce, pe, on="strike", how="outer").sort_values("strike").reset_index(drop=True)
        # Also keep long format cols for compatibility with strategies that use instrument_type/ltp
        wide["_raw"] = None   # marker; raw data accessible via chain_df if needed
        return wide

    def get_live_quote(self, instrument_key: str) -> dict:
        """
        instrument_key: zerodha_symbol like 'NIFTY 50'
        Returns dict with ltp, open, high, low, close, volume.
        """
        kite_key = f"NSE:{instrument_key}"
        try:
            resp = self.kite.quote([kite_key])
            q = resp[kite_key]
            ohlc = q.get("ohlc", {})
            return {
                "ltp":    float(q.get("last_price", 0)),
                "open":   float(ohlc.get("open", 0)),
                "high":   float(ohlc.get("high", 0)),
                "low":    float(ohlc.get("low", 0)),
                "close":  float(ohlc.get("close", 0)),
                "volume": int(q.get("volume", 0)),
            }
        except Exception as e:
            logger.error("get_live_quote failed for %s: %s", instrument_key, e)
            return {"ltp": 0.0, "open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0}

    def get_ltp(self, instrument_keys: List[str]) -> dict:
        """
        Batched LTP fetch. instrument_keys: list of zerodha_symbol strings.
        Returns {key: ltp}.
        """
        kite_keys = [f"NSE:{k}" if ":" not in k else k for k in instrument_keys]
        try:
            resp = self.kite.ltp(kite_keys)
            return {
                k: float(resp.get(f"NSE:{k}", resp.get(k, {})).get("last_price", 0.0))
                for k in instrument_keys
            }
        except Exception as e:
            logger.warning("Batched LTP failed: %s — falling back to per-key.", e)
            return super().get_ltp(instrument_keys)

    def place_order(
        self,
        instrument_key: str,   # tradingsymbol, e.g. "NIFTY2571024750CE"
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float = 0.0,
        tag: str = "",
    ) -> str:
        from kiteconnect import KiteConnect
        exchange = "NFO" if any(x in instrument_key for x in ["CE", "PE"]) else "NSE"
        kite_order_type = (
            KiteConnect.ORDER_TYPE_MARKET if order_type == "MARKET"
            else KiteConnect.ORDER_TYPE_LIMIT
        )
        kite_txn = (
            KiteConnect.TRANSACTION_TYPE_BUY  if transaction_type == "BUY"
            else KiteConnect.TRANSACTION_TYPE_SELL
        )
        order_id = self.kite.place_order(
            variety  = KiteConnect.VARIETY_REGULAR,
            exchange = exchange,
            tradingsymbol   = instrument_key,
            transaction_type= kite_txn,
            quantity        = quantity,
            product         = KiteConnect.PRODUCT_MIS,   # intraday margin for weekly options
            order_type      = kite_order_type,
            price           = price if order_type == "LIMIT" else None,
            tag             = tag[:20] if tag else None,
        )
        logger.info("Order placed: %s %s %s qty=%d order_id=%s",
                    transaction_type, instrument_key, order_type, quantity, order_id)
        return str(order_id)

    def get_positions(self) -> pd.DataFrame:
        try:
            resp = self.kite.positions()
            day_pos = resp.get("day", [])
            return pd.DataFrame(day_pos) if day_pos else pd.DataFrame()
        except Exception as e:
            logger.error("get_positions failed: %s", e)
            return pd.DataFrame()

    def get_order_status(self, order_id: str) -> dict:
        try:
            orders = self.kite.orders()
            for o in orders:
                if str(o["order_id"]) == str(order_id):
                    return o
            return {"status": "NOT_FOUND", "order_id": order_id}
        except Exception as e:
            logger.error("get_order_status failed: %s", e)
            return {"status": "ERROR", "message": str(e)}

    def get_available_margin(self) -> float:
        """Return available cash margin for F&O segment in ₹."""
        try:
            margins = self.kite.margins(segment="equity")  # or "commodity"
            # For F&O: use 'net' from equity segment
            return float(margins.get("net", 0.0))
        except Exception as e:
            logger.error("get_available_margin failed: %s", e)
            return 0.0


# ── Login helper ────────────────────────────────────────────────────
# Run: python -m src.broker.zerodha_login
# This is a separate file to keep zerodha.py clean.
