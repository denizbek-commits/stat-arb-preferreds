import datetime
import itertools
from collections import Counter
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_ticker_data(
    ticker: str,
    start_date: datetime.date,
    end_date: datetime.date
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a ticker between start_date and end_date using yfinance.
    Returns None if empty or on error.
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            print(f"[WARN] No data found for {ticker}.")
            return None
        return data
    except Exception as e:
        print(f"[ERROR] Error fetching data for {ticker}: {e}")
        return None


def analyze_ticker_pairs(
    tickers: List[str],
    start_date: datetime.date,
    end_date: datetime.date,
    output_file: str = "pair_trade_messages.txt",
) -> Dict[str, Any]:
    """
    Analyze all unique pairs of tickers and generate spread/z-score-based trade signals.

    Steps per pair:
        - Fetch weekly adjusted close prices
        - Build spread series (ticker1 - ticker2)
        - Compute correlation, z-score, basic mean-reversion probability
        - Check if spread is near historical MAX or MIN level
        - Score and record trade signals

    Returns:
        Dictionary with pair_signals, summary lists, and position summary text.
    """
    pair_scores: List[Dict[str, Any]] = []
    long_positions: List[str] = []
    short_positions: List[str] = []

    for ticker1, ticker2 in itertools.combinations(tickers, 2):
        data1 = fetch_ticker_data(ticker1, start_date, end_date)
        data2 = fetch_ticker_data(ticker2, start_date, end_date)

        if data1 is None or data2 is None:
            continue

        # Convert to weekly adjusted close
        data1_weekly = data1["Adj Close"].resample("W").last()
        data2_weekly = data2["Adj Close"].resample("W").last()

        aligned = pd.concat([data1_weekly, data2_weekly], axis=1)
        aligned.columns = [
            f"{ticker1} Adj Close",
            f"{ticker2} Adj Close"
        ]
        aligned.dropna(inplace=True)

        if aligned.empty:
            print(f"[INFO] No overlapping weekly data for {ticker1} & {ticker2}.")
            continue

        # Spread
        spread = aligned[f"{ticker1} Adj Close"] - aligned[f"{ticker2} Adj Close"]
        spread_df = pd.DataFrame(
            {
                "Open": spread,
                "High": spread,
                "Low": spread,
                "Close": spread,
            },
            index=aligned.index,
        )

        highest = spread_df["Close"].max()
        lowest = spread_df["Close"].min()
        current = spread_df["Close"].iloc[-1]

        # Correlation
        series1 = aligned[f"{ticker1} Adj Close"]
        series2 = aligned[f"{ticker2} Adj Close"]
        correlation = series1.corr(series2) if len(series1) > 1 else np.nan

        # Z-score
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_score = (current - spread_mean) / spread_std if spread_std != 0 else 0.0

        # Mean reversion probability proxy
        p_reverse = 1 - abs(z_score) / 3
        mean_rev_prob = float(max(min(p_reverse, 1), 0))

        # Score
        total_score = (
            abs(correlation) * 2
            + mean_rev_prob * 3
            + abs(z_score) * 1
        )

        # Signal logic
        is_max = (highest - current) <= 0.05 * abs(highest)
        is_min = (current - lowest) <= 0.05 * abs(lowest)

        if is_max or is_min:
            if is_max:
                signal_type = "Max level"
                risk = highest - current
                reward = current - lowest
                long_positions.append(ticker2)
                short_positions.append(ticker1)
            else:
                signal_type = "Min level"
                risk = current - lowest
                reward = highest - current
                long_positions.append(ticker1)
                short_positions.append(ticker2)

            message = (
                f"{ticker1} & {ticker2} pair reached {signal_type}.\n"
                f"Risk = {risk:.2f}, Reward = {reward:.2f}\n"
                f"Correlation: {correlation:.2f}\n"
                f"Z-Score: {z_score:.2f}\n"
                f"Mean-Reversion Prob.: {mean_rev_prob * 100:.1f}%\n"
                f"Total Score: {total_score:.2f}\n\n"
            )

            pair_scores.append(
                {
                    "message": message,
                    "total_score": float(total_score),
                    "ticker1": ticker1,
                    "ticker2": ticker2,
                    "signal_type": signal_type,
                    "correlation": float(correlation),
                    "z_score": float(z_score),
                    "mean_reversion_prob": float(mean_rev_prob),
                    "risk": float(risk),
                    "reward": float(reward),
                    "spread": spread_df,
                }
            )

    # Sort signals
    pair_scores.sort(key=lambda x: x["total_score"], reverse=True)
    sorted_messages = [pair["message"] for pair in pair_scores]

    # Position summary
    long_counter = Counter(long_positions)
    short_counter = Counter(short_positions)

    sorted_longs = sorted(long_counter.items(), key=lambda x: (-x[1], x[0]))
    sorted_shorts = sorted(short_counter.items(), key=lambda x: (-x[1], x[0]))

    position_summary = (
        "\n" + "=" * 60 + "\n"
        "POSITION SUMMARY\n"
        + "=" * 60 + "\n\n"
        "LONGS:\n"
    )

    if sorted_longs:
        for t, c in sorted_longs:
            position_summary += f"{t} ({c})\n"
    else:
        position_summary += "None\n"

    position_summary += "\nSHORTS:\n"
    if sorted_shorts:
        for t, c in sorted_shorts:
            position_summary += f"{t} ({c})\n"
    else:
        position_summary += "None\n"

    # Write messages to file
    with open(output_file, "w") as f:
        if sorted_messages:
            f.write("PAIR TRADING SIGNALS (SORTED BY SCORE)\n")
            f.write("=" * 60 + "\n\n")
            f.writelines(sorted_messages)
            f.write(position_summary)
        else:
            f.write("No pair signals detected.\n")

    return {
        "pair_signals": pair_scores,
        "longs": sorted_longs,
        "shorts": sorted_shorts,
        "position_summary_text": position_summary,
    }


if __name__ == "__main__":
    tickers_input = input("Enter tickers separated by commas: ")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=5 * 365)

    results = analyze_ticker_pairs(tickers, start_date, end_date)

    if not results["pair_signals"]:
        print("No signals found.")
    else:
        print(f"\nFound {len(results['pair_signals'])} signals.")
        print("Top 5:")
        print("-" * 80)
        for i, pair in enumerate(results["pair_signals"][:5], 1):
            print(f"{i}. {pair['ticker1']} & {pair['ticker2']} — Score: {pair['total_score']:.2f}")
            print(f"   Signal: {pair['signal_type']}, Corr: {pair['correlation']:.2f}, Z: {pair['z_score']:.2f}")
            print()
        print(results["position_summary_text"])
