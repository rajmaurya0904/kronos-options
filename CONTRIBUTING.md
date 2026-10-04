# Contributing to Kronos Options

Thanks for taking a look. Bug reports, ideas and pull requests are all welcome.

## Good first contributions

These are on the [roadmap](README.md#roadmap) and are open for anyone to pick up:

- **Stop-loss and target exits.** Positions currently exit only at square-off time.
- **Zerodha broker.** `src/broker/zerodha.py` is a stub; implement the `BrokerInterface` methods.
- **Longer backtests.** Walk-forward runs across different market regimes, with results.
- **Automatic Upstox token refresh.**

Not sure where to start? Open an issue describing what you'd like to do, and we can talk it through before you write code.

## Setup

```bash
git clone https://github.com/<your-username>/kronos-options.git
cd kronos-options
pip install -r requirements.txt
cp .env.example .env    # on Windows: copy .env.example .env
```

You only need the [Kronos model](https://github.com/shiyu-coder/Kronos) and an Upstox token for forecasting, backtests and paper trading. The unit tests need neither.

## Run the tests

```bash
pytest tests/ -v
```

The tests are fast and never load the model or call Upstox. GitHub Actions runs them on every push and pull request.

## Pull request checklist

- [ ] One focused change per PR, with a clear description of what and why
- [ ] `pytest tests/` passes, and new logic has a test where practical
- [ ] Settings go in `config.yaml`, not hard-coded values
- [ ] **No secrets.** Never commit `.env`, tokens, API keys or account details
- [ ] Changes to trading behaviour are explained in the PR (what it changes and how it was checked)
- [ ] README updated if usage or behaviour changed

## Ground rules for trading code

Correctness matters more than cleverness here, because this code prices trades and can place orders.

- **No lookahead.** A decision at time *t* may only use data available before *t*.
- **Honest pricing.** Prefer real option data; label anything modelled (`bs_approximation`) so it can be filtered.
- **Costs stay in.** Don't remove charges or slippage to make results look better.
- **Live trading stays locked.** Don't weaken the safety gates in `src/live_trader.py`. PRs that do will not be merged.

## Reporting bugs

Open an issue with:

- what you ran (command and config changes)
- what you expected, and what happened instead
- the log output, **with any tokens or account details removed**

## Security

If you find a security problem, such as a way credentials could leak or orders could be placed without the safety gates, please don't open a public issue. Contact the maintainer privately through their [GitHub profile](https://github.com/rajmaurya0904).

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
