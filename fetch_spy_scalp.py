#!/usr/bin/env python3
"""
SPY 0DTE Scalping Data Fetcher
Fetches SPY 5-min intraday data with scalping-specific features.
Trading window: 10:00 - 14:00 ET only.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def compute_intraday_features(df):
    """Compute intraday-specific features for SPY scalping."""
    if len(df) < 50:
        return df

    # --- Standard indicators (on 5-min bars) ---
    # RSI (14-period = 70 min of data)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD (fast on intraday)
    ema8 = df["Close"].ewm(span=8, adjust=False).mean()
    ema21 = df["Close"].ewm(span=21, adjust=False).mean()
    df["macd"] = ema8 - ema21
    df["macd_signal"] = df["macd"].ewm(span=5, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20-period = 100 min)
    bb_sma = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = bb_sma + 2 * bb_std
    df["bb_lower"] = bb_sma - 2 * bb_std
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_sma

    # ATR (14-period)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / df["Close"] * 100

    # Stochastic
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["stoch_k"] = 100 * (df["Close"] - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- INTRADAY-SPECIFIC features ---

    # VWAP
    df["cum_vol"] = df["Volume"].cumsum()
    df["cum_vp"] = (df["Close"] * df["Volume"]).cumsum()
    df["vwap"] = df["cum_vp"] / df["cum_vol"].replace(0, np.nan)
    df["vwap_deviation"] = (df["Close"] - df["vwap"]) / df["vwap"] * 100

    # Distance from day high/low (resets per day)
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert("America/New_York")
        df["day"] = dt.dt.date.astype(str)
    else:
        df["day"] = (df.index // 78).astype(str)

    df["day_high"] = df.groupby("day")["High"].transform("cummax")
    df["day_low"] = df.groupby("day")["Low"].transform("cummin")
    day_range = (df["day_high"] - df["day_low"]).replace(0, np.nan)
    df["day_range_position"] = (df["Close"] - df["day_low"]) / day_range

    # Opening range (first 30 min = 6 bars)
    def opening_range(group):
        first_6 = group.head(6)
        or_high = first_6["High"].max()
        or_low = first_6["Low"].min()
        group["or_high"] = or_high
        group["or_low"] = or_low
        group["above_or"] = (group["Close"] > or_high).astype(int)
        group["below_or"] = (group["Close"] < or_low).astype(int)
        or_range = or_high - or_low if or_high != or_low else 1
        group["or_position"] = (group["Close"] - or_low) / or_range
        return group

    df = df.groupby("day", group_keys=False).apply(opening_range)

    # Time encoding (hour + minute as features)
    if "Date" in df.columns:
        dt = pd.to_datetime(df["Date"])
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert("America/New_York")
        df["hour"] = dt.dt.hour
        df["minute"] = dt.dt.minute
        df["time_pct"] = ((dt.dt.hour - 9) * 60 + dt.dt.minute - 30) / 390  # 0=open, 1=close
    else:
        df["hour"] = 12
        df["minute"] = 0
        df["time_pct"] = 0.5

    # Momentum
    df["return_1"] = df["Close"].pct_change(1)   # last 5 min
    df["return_3"] = df["Close"].pct_change(3)   # last 15 min
    df["return_6"] = df["Close"].pct_change(6)   # last 30 min
    df["return_12"] = df["Close"].pct_change(12) # last 1 hour

    # Volatility (recent)
    df["vol_5"] = df["return_1"].rolling(5).std()   # 25-min vol
    df["vol_12"] = df["return_1"].rolling(12).std()  # 1-hour vol

    # Volume relative to intraday average
    if "day" not in df.columns:
        if "Date" in df.columns:
            dtt = pd.to_datetime(df["Date"])
            if dtt.dt.tz is not None:
                dtt = dtt.dt.tz_convert("America/New_York")
            df["day"] = dtt.dt.date.astype(str)
        else:
            df["day"] = (df.index // 78).astype(str)
    df["vol_sma"] = df.groupby("day")["Volume"].transform(lambda x: x.expanding().mean())
    df["volume_ratio"] = df["Volume"] / df["vol_sma"].replace(0, np.nan)

    # Consecutive candles
    df["candle_dir"] = np.sign(df["Close"] - df["Open"])
    df["consec_up"] = df["candle_dir"].rolling(6).apply(lambda x: (x > 0).sum())
    df["consec_down"] = df["candle_dir"].rolling(6).apply(lambda x: (x < 0).sum())

    # Candle body ratio
    body = (df["Close"] - df["Open"]).abs()
    full_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["body_ratio"] = body / full_range

    # Close in range
    df["close_in_range"] = (df["Close"] - df["Low"]) / full_range

    return df


def create_scalp_labels(df, forward_bars=3, threshold=0.001):
    """
    Label for scalping: will price move threshold% in the next N bars?
    
    Labels:
      1 = UP (price moved up 0.1%+ in next 15 min)
      0 = DOWN (price moved down or sideways)
    """
    future_return = df["Close"].shift(-forward_bars) / df["Close"] - 1

    # Binary: UP or not
    df["label"] = (future_return > threshold).astype(int)
    df["future_return"] = future_return
    return df


def fetch_spy_scalp_data(period="60d", interval="5m", forward_bars=3,
                          threshold=0.001, trading_start=10, trading_end=14):
    """Fetch SPY intraday data for scalping."""
    print(f"📈 Fetching SPY {interval} data...")

    data = yf.download("SPY", period=period, interval=interval, progress=False)
    if data is None or len(data) < 100:
        print("ERROR: No SPY data")
        return None

    data = data.reset_index()
    if hasattr(data.columns, 'levels'):
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

    # Normalize date column
    date_col = "Datetime" if "Datetime" in data.columns else "Date"
    if date_col == "Datetime":
        data = data.rename(columns={"Datetime": "Date"})

    data["Date"] = pd.to_datetime(data["Date"])
    data["ticker"] = "SPY"

    print(f"   Raw data: {len(data)} bars")
    print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")

    # Compute features
    data = compute_intraday_features(data)

    # Create labels
    data = create_scalp_labels(data, forward_bars=forward_bars, threshold=threshold)

    # Filter to trading window only (10:00 - 14:00 ET)
    dt_filter = pd.to_datetime(data["Date"])
    if dt_filter.dt.tz is not None:
        dt_filter = dt_filter.dt.tz_convert("America/New_York")
    data["hour_val"] = dt_filter.dt.hour
    before = len(data)
    data = data[(data["hour_val"] >= trading_start) & (data["hour_val"] < trading_end)]
    print(f"   After {trading_start}:00-{trading_end}:00 filter: {len(data)} bars (was {before})")

    # Drop NaN
    feature_cols = [c for c in data.columns if c not in 
                    ["Date", "ticker", "label", "future_return", "day", "hour_val",
                     "cum_vol", "cum_vp", "or_high", "or_low", "day_high", "day_low",
                     "vol_sma", "extreme_type"]]
    data = data.dropna(subset=["label"])
    data = data.dropna(subset=feature_cols, how="any")

    # Save
    out_path = os.path.join(DATA_DIR, "spy_scalp_data.csv")
    data.to_csv(out_path, index=False)

    # Stats
    ups = data["label"].sum()
    total = len(data)
    print(f"\n✅ SPY Scalp Data:")
    print(f"   Bars: {total}")
    print(f"   UP signals: {int(ups)} ({ups/total*100:.1f}%)")
    print(f"   DOWN signals: {int(total - ups)} ({(total-ups)/total*100:.1f}%)")
    print(f"   Forward: {forward_bars} bars ({forward_bars * 5} min)")
    print(f"   Threshold: {threshold * 100:.2f}%")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Saved: {out_path}")

    meta = {
        "bars": total,
        "up_pct": round(ups / total * 100, 1),
        "forward_bars": forward_bars,
        "threshold": threshold,
        "interval": interval,
        "trading_hours": f"{trading_start}:00-{trading_end}:00",
        "features": len(feature_cols),
        "feature_names": feature_cols,
        "date_range": [str(data["Date"].min()), str(data["Date"].max())],
    }
    with open(os.path.join(DATA_DIR, "spy_scalp_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SPY 0DTE Scalp Data Fetcher")
    parser.add_argument("--interval", default="5m", help="Bar interval (default: 5m)")
    parser.add_argument("--period", default="60d", help="Data period (default: 60d)")
    parser.add_argument("--forward", type=int, default=3, help="Forward bars for label (default: 3 = 15 min)")
    parser.add_argument("--threshold", type=float, default=0.001, help="Min move for UP label (default: 0.001 = 0.1%%)")
    parser.add_argument("--start-hour", type=int, default=10, help="Trading window start (default: 10)")
    parser.add_argument("--end-hour", type=int, default=14, help="Trading window end (default: 14)")
    args = parser.parse_args()

    print(f"📈 SPY 0DTE Scalp Data Fetcher")
    print(f"   Window: {args.start_hour}:00 - {args.end_hour}:00 ET")
    print(f"   Forward: {args.forward} bars ({args.forward * 5} min)")
    print(f"   Threshold: {args.threshold * 100:.2f}%")
    print("=" * 50)

    fetch_spy_scalp_data(
        period=args.period,
        interval=args.interval,
        forward_bars=args.forward,
        threshold=args.threshold,
        trading_start=args.start_hour,
        trading_end=args.end_hour,
    )
