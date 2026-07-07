"""Quick backtest runner — NIFTY May-June 2026 with real Zerodha/Kite data."""
import logging
import warnings
warnings.filterwarnings("ignore")

import src.utils as u
u._CONFIG_CACHE = None  # force fresh config read

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for name in ["urllib3", "requests", "transformers", "huggingface_hub", "filelock"]:
    logging.getLogger(name).setLevel(logging.ERROR)

from src.utils import get_broker
from src.backtester import Backtester

broker = get_broker()
bt     = Backtester(broker=broker)

print()
print("=" * 55)
print("  KRONOS NIFTY BACKTEST  2026-05-01 -> 2026-06-09")
print("=" * 55)
print()

result = bt.run("NIFTY", "2026-05-01", "2026-06-09", lots=1)

stats = result["stats"]
tl    = result["trade_log"]

print()
print("=" * 55)
print("  RESULTS")
print("=" * 55)
print("  Trades:       ", stats["total_trades"], " (", stats.get("wins", 0), "W /", stats.get("losses", 0), "L)")
print("  Win rate:     ", round(stats["win_rate_pct"], 1), "%")
print("  Total P&L:    Rs", round(stats["total_pnl_rs"], 0))
print("  Avg win:      Rs", round(stats.get("avg_win_rs", 0), 0))
print("  Avg loss:     Rs", round(stats.get("avg_loss_rs", 0), 0))
print("  Profit factor:", round(stats["profit_factor"], 2))
print("  Sharpe:       ", round(stats["sharpe"], 2))
print("  Max drawdown: Rs", round(stats["max_drawdown_rs"], 0))
print("=" * 55)

if not tl.empty:
    cols = ["trade_date", "signal", "strategy", "entry_price", "exit_price", "net_pnl_rs", "data_source"]
    keep = [c for c in cols if c in tl.columns]
    print()
    print(tl[keep].to_string(index=False))
