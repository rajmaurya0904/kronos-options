"""
Screenshot every page of the running Streamlit dashboard.

    streamlit run dashboard/app.py --server.headless true &
    python scripts/capture_dashboard.py http://localhost:8501 docs/screenshots

Used by .github/workflows/screenshots.yml; needs `pip install playwright`
and `playwright install chromium`.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGES = {
    "Live Forecasts": "live-forecasts",
    "Today's Signals": "todays-signals",
    "Paper P&L": "paper-pnl",
    "Backtest Results": "backtest-results",
}


def main(url: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Tall viewport: Streamlit scrolls inside its own container, so a
        # full-page screenshot only captures what fits the viewport.
        page = browser.new_page(viewport={"width": 1440, "height": 1500}, device_scale_factor=2)
        page.goto(url, wait_until="networkidle")
        page.get_by_text("Kronos Options", exact=True).first.wait_for(timeout=60_000)
        for label, slug in PAGES.items():
            page.locator('[data-testid="stSidebar"]').get_by_text(label, exact=True).click()
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(4000)  # let Plotly finish drawing
            page.screenshot(path=str(out / f"{slug}.png"))
            print("saved", out / f"{slug}.png")
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
