#!/usr/bin/env python3
"""
Mean Reversion Data Fetcher — Different labels for reversion prediction.
Instead of "will price go up/down tomorrow?" we ask:
"After an extreme move, will price revert within N days?"

Labels:
  1 = Price reverted (bounced back 1%+ after a drop, or pulled back 1%+ after a spike)
  0 = Price did NOT revert (continued in the same direction)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from fetch_data import TICKERS, compute_indicators

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def create_reversion_labels(df, rsi_oversold=30, rsi_overbought=70, 
                             bb_threshold=0.1, reversion_target=0.01,
                             lookahead=3):
    """
    Create mean reversion labels.
    
    Only labels rows where an EXTREME condition exists:
    - RSI < oversold OR RSI > overbought
    - OR price outside Bollinger Bands (bb_position < bb_threshold or > 1-bb_threshold)
    
    Then checks: did price revert by reversion_target% within lookahead days?
    
    Args:
        rsi_oversold: RSI level for oversold (default 30)
        rsi_overbought: RSI level for overbought (default 70) 
        bb_threshold: How far outside BB to trigger (0.1 = below lower 10%)
        reversion_target: Required reversion % (0.01 = 1%)
        lookahead: Days to wait for reversion
    """
    labels = pd.Series(np.nan, index=df.index)  # NaN = no extreme, skip
    future_returns = pd.Series(np.nan, index=df.index)
    extreme_type = pd.Series("none", index=df.index)
    
    for i in range(len(df) - lookahead):
        rsi = df["rsi"].iloc[i] if "rsi" in df.columns else 50
        bb_pos = df["bb_position"].iloc[i] if "bb_position" in df.columns else 0.5
        current_price = df["Close"].iloc[i]
        
        is_oversold = rsi < rsi_oversold or bb_pos < bb_threshold
        is_overbought = rsi > rsi_overbought or bb_pos > (1 - bb_threshold)
        
        if not (is_oversold or is_overbought):
            continue  # No extreme condition, skip
        
        # Look ahead N days for reversion
        future_prices = df["Close"].iloc[i+1:i+1+lookahead]
        
        if is_oversold:
            # Expecting a bounce UP
            max_future = future_prices.max()
            reversion_pct = (max_future - current_price) / current_price
            labels.iloc[i] = 1 if reversion_pct >= reversion_target else 0
            future_returns.iloc[i] = reversion_pct
            extreme_type.iloc[i] = "oversold"
            
        elif is_overbought:
            # Expecting a pullback DOWN
            min_future = future_prices.min()
            reversion_pct = (current_price - min_future) / current_price
            labels.iloc[i] = 1 if reversion_pct >= reversion_target else 0
            future_returns.iloc[i] = reversion_pct
            extreme_type.iloc[i] = "overbought"
    
    df["label"] = labels
    df["future_return"] = future_returns
    df["extreme_type"] = extreme_type
    return df


def fetch_mean_reversion_data(tickers=None, period="5y", interval="1d",
                               rsi_oversold=30, rsi_overbought=70,
                               reversion_target=0.01, lookahead=3):
    """Fetch and process data for mean reversion."""
    if tickers is None:
        tickers = TICKERS
    
    all_data = []
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{total}] {ticker}...")
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data is None or len(data) < 100:
                continue
            data = data.reset_index()
            if hasattr(data.columns, 'levels'):
                data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
            if "Datetime" in data.columns and "Date" not in data.columns:
                data = data.rename(columns={"Datetime": "Date"})
            data["ticker"] = ticker
            
            # Compute indicators
            data = compute_indicators(data)
            
            # Create reversion labels
            data = create_reversion_labels(
                data, 
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                reversion_target=reversion_target,
                lookahead=lookahead
            )
            
            all_data.append(data)
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(0.1)
    
    if not all_data:
        print("ERROR: No data!")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Only keep rows with extreme conditions (label is not NaN)
    before = len(combined)
    combined = combined.dropna(subset=["label"])
    after = len(combined)
    
    print(f"\n📊 Extreme conditions found: {after} out of {before} total rows ({after/before*100:.1f}%)")
    
    # Drop NaN indicators
    indicator_cols = [c for c in combined.columns if c not in ["Date", "ticker", "label", "future_return", "extreme_type"]]
    combined = combined.dropna(subset=indicator_cols, how="any")
    
    # Save
    out_path = os.path.join(DATA_DIR, "mean_reversion_data.csv")
    combined.to_csv(out_path, index=False)
    
    # Stats
    reverted = combined["label"].sum()
    total_extremes = len(combined)
    oversold = len(combined[combined["extreme_type"] == "oversold"])
    overbought = len(combined[combined["extreme_type"] == "overbought"])
    
    meta = {
        "total_rows": total_extremes,
        "reverted": int(reverted),
        "not_reverted": int(total_extremes - reverted),
        "reversion_rate": round(reverted / total_extremes * 100, 1) if total_extremes > 0 else 0,
        "oversold_count": oversold,
        "overbought_count": overbought,
        "rsi_oversold": rsi_oversold,
        "rsi_overbought": rsi_overbought,
        "reversion_target": reversion_target,
        "lookahead": lookahead,
        "period": period,
        "interval": interval,
        "tickers": len(tickers),
        "fetched_at": datetime.now().isoformat(),
    }
    
    with open(os.path.join(DATA_DIR, "mean_reversion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Mean Reversion Data:")
    print(f"   Extreme conditions: {total_extremes}")
    print(f"   Reverted: {int(reverted)} ({meta['reversion_rate']}%)")
    print(f"   Oversold: {oversold} | Overbought: {overbought}")
    print(f"   RSI thresholds: <{rsi_oversold} / >{rsi_overbought}")
    print(f"   Reversion target: {reversion_target*100}% within {lookahead} days")
    print(f"   Saved: {out_path}")
    
    return combined


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mean Reversion Data Fetcher")
    parser.add_argument("--period", default="5y", help="Data period (default: 5y)")
    parser.add_argument("--interval", default="1d", help="Candle interval (default: 1d)")
    parser.add_argument("--rsi-low", type=int, default=30, help="RSI oversold threshold (default: 30)")
    parser.add_argument("--rsi-high", type=int, default=70, help="RSI overbought threshold (default: 70)")
    parser.add_argument("--target", type=float, default=0.01, help="Reversion target %% (default: 0.01 = 1%%)")
    parser.add_argument("--lookahead", type=int, default=3, help="Days to wait for reversion (default: 3)")
    args = parser.parse_args()
    
    if args.interval in ["15m", "5m"]:
        args.period = "60d"
    elif args.interval == "1h" and args.period in ["5y", "max"]:
        args.period = "2y"
    
    print(f"🔄 Mean Reversion Data Fetcher")
    print(f"   RSI: <{args.rsi_low} (oversold) / >{args.rsi_high} (overbought)")
    print(f"   Target: {args.target*100}% reversion within {args.lookahead} candles")
    print(f"   Interval: {args.interval} | Period: {args.period}")
    print("=" * 50)
    
    fetch_mean_reversion_data(
        period=args.period,
        interval=args.interval,
        rsi_oversold=args.rsi_low,
        rsi_overbought=args.rsi_high,
        reversion_target=args.target,
        lookahead=args.lookahead,
    )
