#!/usr/bin/env python3
"""
Backtest: Compare current Opscan v1.1 scoring weights vs Neural Net-informed weights.
Uses the same historical data and same methodology — just different factor weights.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_data():
    """Load training data with all indicators."""
    path = os.path.join(DATA_DIR, "training_data.csv")
    if not os.path.exists(path):
        print("❌ No training data. Run fetch_data.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    return df


# ── Opscan v1.1 Current Weights ──
# Maps our neural net features to Opscan factor concepts
OPSCAN_V11 = {
    "name": "Opscan v1.1 (Current)",
    "weights": {
        "rsi": 12,           # F4: RSI
        "volume_ratio": 15,  # F6: Unusual Volume
        "adx": 8,            # F2: Trend Strength (proxy)
        "bb_position": 10,   # F3: IV Rank (proxy via BB position)
        "macd": 10,          # F1: Directional Alignment
        "daily_range_pct": 8,# F11: Range Position
        "atr_pct": 5,        # Not a direct factor (underweighted)
        "return_1d": 6,      # F8: Momentum (proxy)
        "stoch_k": 6,        # Part of RSI family
        "above_sma20": 8,    # F2: Trend
        "above_sma50": 8,    # F2: Trend
        "close_in_range": 4, # F12: Spread gate (proxy)
    },
}

# ── Neural Net Informed Weights v1.0 ──
# Weights derived from neural net feature importance (500K samples)
NEURAL_V1 = {
    "name": "Neural Net Informed v1.0",
    "weights": {
        "atr_pct": 20,           # #1 feature — massively boosted
        "volatility_20d": 15,    # #2 — new factor
        "volume_ratio": 15,      # #3 — confirmed important
        "return_1d": 12,         # #4 — momentum boosted
        "daily_range_pct": 10,   # #5 — volatility
        "close_in_range": 10,    # #6 — range position
        "return_10d": 8,         # #7 — medium-term momentum
        "plus_di": 6,            # #8 — directional
        "macd": 5,               # demoted from v1.1
        "above_sma20": 4,        # demoted
        "adx": 3,                # demoted
        "rsi": 2,                # nearly removed — dead last in importance
    },
}

# ── Neural Net Top-Heavy v1.1 ──
# Only use top 5 features, ignore the rest
NEURAL_TOP5 = {
    "name": "Neural Net Top-5 Only",
    "weights": {
        "atr_pct": 30,
        "volatility_20d": 25,
        "volume_ratio": 20,
        "return_1d": 15,
        "daily_range_pct": 10,
    },
}


def compute_composite_score(row, weights):
    """Compute weighted composite score for a single row."""
    score = 0
    max_possible = 0
    
    for feature, weight in weights.items():
        val = row.get(feature, np.nan)
        if pd.isna(val):
            continue
        
        # Normalize feature to 0-1 range using simple percentile-like scaling
        # This is a simplified version — production would use proper scaling
        score += val * weight
        max_possible += abs(weight)
    
    if max_possible == 0:
        return 50
    
    return score


def backtest_strategy(df, config, threshold_pct=70):
    """
    Backtest a scoring configuration.
    
    Logic:
    - Compute composite score for each row
    - If score > threshold percentile → predict BUY
    - If score < (100-threshold) percentile → predict SELL
    - Otherwise → NEUTRAL (no trade)
    - Check if prediction matches actual label
    """
    weights = config["weights"]
    name = config["name"]
    
    # Compute scores
    scores = df.apply(lambda row: compute_composite_score(row, weights), axis=1)
    df = df.copy()
    df["composite_score"] = scores
    
    # Remove NaN scores
    df = df.dropna(subset=["composite_score", "label", "future_return"])
    
    # Use percentile thresholds
    buy_thresh = np.percentile(scores.dropna(), threshold_pct)
    sell_thresh = np.percentile(scores.dropna(), 100 - threshold_pct)
    
    # Generate signals
    df["signal"] = 0  # neutral
    df.loc[df["composite_score"] >= buy_thresh, "signal"] = 1   # buy
    df.loc[df["composite_score"] <= sell_thresh, "signal"] = -1  # sell
    
    # Split into train/test (last 20% = test)
    split = int(len(df) * 0.8)
    test = df.iloc[split:].copy()
    
    # Evaluate
    total_signals = len(test[test["signal"] != 0])
    buy_signals = len(test[test["signal"] == 1])
    sell_signals = len(test[test["signal"] == -1])
    
    if total_signals == 0:
        return {"name": name, "error": "No signals generated"}
    
    # Buy accuracy: did price actually go up?
    buys = test[test["signal"] == 1]
    buy_correct = len(buys[buys["future_return"] > 0]) if len(buys) > 0 else 0
    buy_accuracy = buy_correct / len(buys) * 100 if len(buys) > 0 else 0
    
    # Sell accuracy: did price actually go down?
    sells = test[test["signal"] == -1]
    sell_correct = len(sells[sells["future_return"] < 0]) if len(sells) > 0 else 0
    sell_accuracy = sell_correct / len(sells) * 100 if len(sells) > 0 else 0
    
    # Combined accuracy
    correct = buy_correct + sell_correct
    combined_accuracy = correct / total_signals * 100
    
    # Simulated P/L (assuming equal position size)
    buy_returns = buys["future_return"].sum() if len(buys) > 0 else 0
    sell_returns = -sells["future_return"].sum() if len(sells) > 0 else 0  # short = inverse
    total_return = buy_returns + sell_returns
    avg_return = total_return / total_signals * 100 if total_signals > 0 else 0
    
    # Win rate
    wins = len(buys[buys["future_return"] > 0]) + len(sells[sells["future_return"] < 0])
    win_rate = wins / total_signals * 100
    
    # Profit factor
    gross_profit = buys[buys["future_return"] > 0]["future_return"].sum() + \
                   (-sells[sells["future_return"] < 0]["future_return"]).sum()
    gross_loss = abs(buys[buys["future_return"] < 0]["future_return"].sum()) + \
                 abs((-sells[sells["future_return"] > 0]["future_return"]).sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Sharpe-like ratio
    all_trade_returns = pd.concat([
        buys["future_return"],
        -sells["future_return"]
    ]) if len(sells) > 0 else buys["future_return"]
    sharpe = (all_trade_returns.mean() / all_trade_returns.std()) * np.sqrt(252) if all_trade_returns.std() > 0 else 0
    
    return {
        "name": name,
        "total_signals": total_signals,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "buy_accuracy": round(buy_accuracy, 1),
        "sell_accuracy": round(sell_accuracy, 1),
        "combined_accuracy": round(combined_accuracy, 1),
        "win_rate": round(win_rate, 1),
        "avg_return_pct": round(avg_return, 3),
        "total_return_pct": round(total_return * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe": round(sharpe, 2),
    }


if __name__ == "__main__":
    print("🔬 Backtest: Scoring Engine Comparison")
    print("=" * 60)
    
    df = load_data()
    print(f"Data: {len(df)} rows")
    print()
    
    configs = [OPSCAN_V11, NEURAL_V1, NEURAL_TOP5]
    results = []
    
    for config in configs:
        result = backtest_strategy(df, config)
        results.append(result)
        
        print(f"📊 {result['name']}")
        print(f"   Signals: {result.get('total_signals', 0)} ({result.get('buy_signals', 0)} buy / {result.get('sell_signals', 0)} sell)")
        print(f"   Win Rate: {result.get('win_rate', 0)}%")
        print(f"   Buy Accuracy: {result.get('buy_accuracy', 0)}%")
        print(f"   Sell Accuracy: {result.get('sell_accuracy', 0)}%")
        print(f"   Profit Factor: {result.get('profit_factor', 0)}")
        print(f"   Sharpe Ratio: {result.get('sharpe', 0)}")
        print(f"   Total Return: {result.get('total_return_pct', 0)}%")
        print()
    
    # Summary comparison
    print("=" * 60)
    print("📋 HEAD-TO-HEAD COMPARISON")
    print(f"{'Metric':<20} ", end="")
    for r in results:
        print(f"{r['name'][:18]:<20} ", end="")
    print()
    print("-" * 80)
    
    for metric in ["win_rate", "profit_factor", "sharpe", "total_return_pct"]:
        print(f"{metric:<20} ", end="")
        values = [r.get(metric, 0) for r in results]
        best = max(values)
        for v in values:
            marker = " 🏆" if v == best and v > 0 else ""
            print(f"{v:<20}{marker} ", end="")
        print()
    
    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "backtest_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
