# Statistical Arbitrage on Preferred Stocks

This project implements a statistical arbitrage framework on preferred stocks and related ETFs
(e.g. PFF) using spread-based mean reversion.

The core ideas:
- Build spreads between pairs/baskets of preferreds and benchmarks
- Compute rolling z-scores of those spreads
- Generate trading signals based on extreme deviations
- Backtest the strategy and evaluate its performance

## Project Structure

- `notebooks/01_exploration.ipynb`  
  Example usage & visualization.

- `src/pairs_scanner.py`  
  Main pair analysis module (functions + CLI entry).


## How to Run

1. Create a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
