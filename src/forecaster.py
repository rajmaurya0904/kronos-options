"""
Kronos forecaster — wraps the Kronos financial OHLCV foundation model
(https://github.com/shiyu-coder/Kronos) for probabilistic price forecasting.

Setup:
    git clone https://github.com/shiyu-coder/Kronos <KRONOS_REPO_PATH>
    cd <KRONOS_REPO_PATH> && pip install -r requirements.txt
    Set KRONOS_REPO_PATH in .env or kronos.repo_path in config.yaml.

Models download from HuggingFace on first use:
    NeoQuasar/Kronos-Tokenizer-base  (~small)
    NeoQuasar/Kronos-small           (24.7M params, 512-token context)

Usage:
    forecaster = KronosForecaster()
    result = forecaster.forecast("NIFTY", df_ohlcv, pred_len=12, samples=20)
"""
from __future__ import annotations
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.db import get_cached_forecast, init_db, make_forecast_hash, save_forecast
from src.utils import load_config, IST

logger = logging.getLogger(__name__)

# ── Pre-flight torch check (emit ONE warning at import time, never per-bar) ──
_TORCH_AVAILABLE = False
_TORCH_WARN_EMITTED = False

try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _warn_torch_once() -> None:
    """Emit the torch-missing warning exactly once per process."""
    global _TORCH_WARN_EMITTED
    if not _TORCH_WARN_EMITTED:
        logger.warning(
            "PyTorch not found — Kronos AI forecasting is DISABLED. "
            "Backtester will fall back to Black-Scholes signals.\n"
            "  Fix: pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu\n"
            "  Then:  cd '%s' && pip install -r requirements.txt",
            os.environ.get("KRONOS_REPO_PATH", "<KRONOS_REPO_PATH>"),
        )
        _TORCH_WARN_EMITTED = True


class KronosForecaster:
    """
    Wrapper around KronosPredictor for probabilistic OHLCV forecasting.

    Calls predict() `samples` times with temperature sampling to build a
    distribution of future paths. Each call is stochastic (T=1.0, top_p<1).
    Model is loaded lazily on first forecast call.

    If PyTorch is unavailable the forecaster raises RuntimeError on the first
    call to forecast(), which the backtester catches and handles via its
    Black-Scholes fallback — no per-bar log spam.
    """

    def __init__(self, config_path: str = "config.yaml", db_path: str = "data/kronos_options.db"):
        from dotenv import load_dotenv
        load_dotenv()

        self.cfg = load_config(config_path)
        self.db_path = db_path
        self._predictor = None
        self._unavailable = False  # set True after first load failure to skip retries
        init_db(db_path)

        kcfg = self.cfg.get("kronos", {})
        kronos_path = kcfg.get("repo_path") or os.environ.get("KRONOS_REPO_PATH", "")
        if not kronos_path:
            raise EnvironmentError(
                "Kronos repo path not set.\n"
                "  1. Clone: git clone https://github.com/shiyu-coder/Kronos <path>\n"
                "  2. Install: cd <path> && pip install -r requirements.txt\n"
                "  3. Set KRONOS_REPO_PATH=<path> in .env  OR  kronos.repo_path in config.yaml"
            )
        self.kronos_path = kronos_path

    # ── Model loading ──────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load Kronos-small + Tokenizer. Called once on first forecast."""
        if self._predictor is not None:
            return
        if self._unavailable:
            raise RuntimeError("Kronos unavailable (previous load failed — see earlier warning).")

        # Gate on torch before attempting any imports
        if not _TORCH_AVAILABLE:
            _warn_torch_once()
            self._unavailable = True
            raise RuntimeError(
                "PyTorch not installed. Install with:\n"
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            )

        logger.info(
            "Loading Kronos-small from %s (first call — downloads models from HuggingFace)…",
            self.kronos_path,
        )

        if self.kronos_path not in sys.path:
            sys.path.insert(0, self.kronos_path)

        try:
            from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore[import]
        except ImportError as e:
            self._unavailable = True
            raise ImportError(
                f"Cannot import Kronos from {self.kronos_path}.\n"
                f"Run: cd {self.kronos_path} && pip install -r requirements.txt\n"
                f"Error: {e}"
            )

        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        self._predictor = KronosPredictor(model, tokenizer, max_context=512)

        logger.info("Kronos-small loaded.")

    # ── Forecast ───────────────────────────────────────────────────────

    def forecast(
        self,
        symbol: str,
        df: pd.DataFrame,
        lookback: Optional[int] = None,
        pred_len: Optional[int] = None,
        samples: Optional[int] = None,
        use_cache: bool = True,
    ) -> dict:
        """
        Generate probabilistic OHLCV forecast by running Kronos `samples` times.

        Args:
            symbol:   Instrument name (NIFTY / BANKNIFTY / SENSEX)
            df:       Clean 5-min OHLCV DataFrame with IST DatetimeIndex
            lookback: Number of input bars (default: from config; Kronos auto-truncates at 512 tokens)
            pred_len: Forecast horizon in bars (default: from config)
            samples:  Number of stochastic paths to generate (default: from config)

        Returns dict:
            median_path:       pd.DataFrame of predicted OHLCV (pred_len rows)
            sample_paths:      np.ndarray shape (samples, pred_len) — predicted close prices
            prob_up:           float, fraction of samples where final close > current close
            expected_move_pct: float, % move from current close to median predicted close
            path_dispersion:   float, std of sample endpoints as % of current close
            expected_range:    (p10_close, p90_close) — 10th/90th percentile of sample endpoints
            expected_close:    float, median predicted final close
            current_close:     float, last known close
        """
        kcfg     = self.cfg.get("kronos", {})
        lookback = lookback or kcfg.get("lookback", 400)
        pred_len = pred_len or kcfg.get("pred_len", 12)
        samples  = samples  or kcfg.get("samples", 20)

        if len(df) < lookback:
            logger.warning(
                "%s: only %d bars available (need %d for full lookback). Using all.",
                symbol, len(df), lookback,
            )

        input_df      = df.tail(lookback).copy()
        last_ts       = str(input_df.index[-1])
        current_close = float(input_df["close"].iloc[-1])

        # Cache check
        input_hash = make_forecast_hash(symbol, last_ts, lookback, pred_len, samples)
        if use_cache and kcfg.get("cache_forecasts", True):
            cached = get_cached_forecast(
                input_hash,
                ttl_minutes=kcfg.get("cache_ttl_minutes", 5),
                db_path=self.db_path,
            )
            if cached:
                logger.debug("Cache hit for %s (hash %s…)", symbol, input_hash[:8])
                return self._deserialise_cached(cached, current_close)

        self._load_model()  # raises RuntimeError if torch/Kronos unavailable

        sample_paths = self._run_kronos(input_df, pred_len, samples)

        # ── Summary statistics ─────────────────────────────────────────
        endpoints          = sample_paths[:, -1]
        median_path_closes = np.median(sample_paths, axis=0)
        median_close       = float(np.median(endpoints))

        prob_up            = float((endpoints > current_close).mean())
        expected_move_pct  = float((median_close - current_close) / current_close * 100)
        path_dispersion    = float(np.std(endpoints) / current_close * 100)
        p10                = float(np.percentile(endpoints, 10))
        p90                = float(np.percentile(endpoints, 90))

        pred_index = pd.date_range(
            start=input_df.index[-1] + pd.Timedelta(minutes=5),
            periods=pred_len,
            freq="5min",
            tz=IST,
        )
        median_path_df = pd.DataFrame({"close": median_path_closes}, index=pred_index)

        # Convert index to ISO strings so the records are JSON-serialisable
        median_path_records = json.loads(median_path_df.reset_index().to_json(orient="records"))

        result = {
            "symbol":            symbol,
            "input_hash":        input_hash,
            "last_bar_ts":       last_ts,
            "created_at":        datetime.now().isoformat(),
            "pred_len":          pred_len,
            "samples":           samples,
            "median_path":       median_path_records,
            "sample_paths":      sample_paths.tolist(),
            "prob_up":           prob_up,
            "expected_move_pct": expected_move_pct,
            "path_dispersion":   path_dispersion,
            "expected_close":    median_close,
            "current_close":     current_close,
            "expected_range":    (p10, p90),
        }

        if kcfg.get("cache_forecasts", True):
            fid = save_forecast(result, db_path=self.db_path)
            result["forecast_id"] = fid

        result["median_path"]  = median_path_df
        result["sample_paths"] = sample_paths
        return result

    def _run_kronos(self, input_df: pd.DataFrame, pred_len: int, samples: int) -> np.ndarray:
        """
        Run Kronos `samples` times with temperature sampling.

        Kronos's sample_count > 1 averages paths, so we call predict() once per
        desired sample to get a true distribution of futures.

        Returns np.ndarray of shape (samples, pred_len) — close price paths.
        """
        kcfg        = self.cfg.get("kronos", {})
        temperature = kcfg.get("temperature", 1.0)
        top_p       = kcfg.get("top_p", 0.9)

        # Build input DataFrame (Kronos expects open/high/low/close + optional volume/amount)
        cols = ["open", "high", "low", "close"]
        if "volume" in input_df.columns:
            cols.append("volume")
        x_df = input_df[cols].copy()

        x_timestamp = pd.Series(input_df.index)
        last_ts     = input_df.index[-1]
        y_timestamp = pd.Series(pd.date_range(
            start=last_ts + pd.Timedelta(minutes=5),
            periods=pred_len,
            freq="5min",
            tz=IST,
        ))

        all_paths = []
        for i in range(samples):
            try:
                pred_df = self._predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=1,
                )
                close_vals = pred_df["close"].values[:pred_len].astype(float)
                all_paths.append(close_vals)
            except Exception as e:
                logger.warning("Kronos sample %d/%d failed: %s — skipping", i + 1, samples, e)

        if not all_paths:
            raise RuntimeError("All Kronos samples failed.")

        return np.array(all_paths)  # (samples, pred_len)

    # ── Deserialise cached result ──────────────────────────────────────

    def _deserialise_cached(self, row: dict, current_close: float) -> dict:
        median_records = json.loads(row["median_path"])
        median_df      = pd.DataFrame(median_records)
        if "index" in median_df.columns:
            median_df = median_df.rename(columns={"index": "timestamp"})
            median_df["timestamp"] = pd.to_datetime(median_df["timestamp"])
            median_df.set_index("timestamp", inplace=True)

        sample_paths = np.array(json.loads(row["sample_paths"]))
        endpoints    = sample_paths[:, -1]
        p10 = float(np.percentile(endpoints, 10))
        p90 = float(np.percentile(endpoints, 90))

        return {
            "symbol":            row["symbol"],
            "input_hash":        row["input_hash"],
            "last_bar_ts":       row["last_bar_ts"],
            "created_at":        row["created_at"],
            "pred_len":          row["pred_len"],
            "samples":           row["samples"],
            "median_path":       median_df,
            "sample_paths":      sample_paths,
            "prob_up":           row["prob_up"],
            "expected_move_pct": row["expected_move_pct"],
            "path_dispersion":   row["path_dispersion"],
            "expected_close":    row["expected_close"],
            "current_close":     current_close,
            "expected_range":    (p10, p90),
            "forecast_id":       row["id"],
        }
