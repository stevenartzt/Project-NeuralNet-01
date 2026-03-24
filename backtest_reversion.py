#!/usr/bin/env python3
"""
Backtest: Mean Reversion Strategy
Uses neural net findings: volatility + volume context at RSI extremes.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def run_reversion_backtest(
    rsi_oversold=30, rsi_overbought=70,
    vol_filter=True, volume_filter=True, invert_volume=False,
    take_profit=0.02, stop_loss=0.03,
    lookahead=5, period_months=6
):
    """
    Backtest mean reversion: enter at RSI extremes, exit on reversion or stop.
    
    Neural net-informed filters:
    - vol_filter: only trade high-volatility stocks (top feature)
    - volume_filter: only trade with above-average volume (top 3 feature)
    """
    path = os.path.join(DATA_DIR, "mean_reversion_data.csv")
    if not os.path.exists(path):
        # Fall back to regular training data
        path = os.path.join(DATA_DIR, "training_data.csv")
    
    df = pd.read_csv(path)
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        if period_months:
            cutoff = df["Date"].max() - pd.Timedelta(days=period_months * 30)
            df = df[df["Date"] >= cutoff]
    
    print(f"Data: {len(df)} rows")
    
    # Find extreme conditions
    trades = []
    
    for ticker in df["ticker"].unique() if "ticker" in df.columns else ["ALL"]:
        if ticker == "ALL":
            tdf = df.copy()
        else:
            tdf = df[df["ticker"] == ticker].sort_values("Date" if "Date" in df.columns else df.columns[0]).reset_index(drop=True)
        
        if len(tdf) < 20:
            continue
        
        for i in range(len(tdf) - lookahead):
            row = tdf.iloc[i]
            rsi = row.get("rsi", 50)
            if pd.isna(rsi):
                continue
            
            # Check for extreme condition
            is_oversold = rsi < rsi_oversold
            is_overbought = rsi > rsi_overbought
            
            if not (is_oversold or is_overbought):
                continue
            
            # Neural net filter 1: Volatility (top feature)
            if vol_filter:
                atr_pct = row.get("atr_pct", 0)
                vol_20d = row.get("volatility_20d", 0)
                if pd.isna(atr_pct) or pd.isna(vol_20d):
                    continue
                # Require above-median volatility
                if atr_pct < tdf["atr_pct"].median() and vol_20d < tdf.get("volatility_20d", pd.Series([0])).median():
                    continue
            
            # Neural net filter 2: Volume
            if volume_filter:
                vol_ratio = row.get("volume_ratio", 1)
                if pd.isna(vol_ratio):
                    continue
                if invert_volume:
                    # LOW volume = quiet dip, more likely to revert
                    if vol_ratio > 1.0:
                        continue
                else:
                    # HIGH volume (original)
                    if vol_ratio < 1.2:
                        continue
            
            entry_price = row.get("Close", 0)
            if entry_price <= 0:
                continue
            
            # Simulate the trade over lookahead days
            direction = "LONG" if is_oversold else "SHORT"
            exit_price = entry_price
            exit_reason = "expired"
            exit_day = lookahead
            
            for j in range(1, min(lookahead + 1, len(tdf) - i)):
                future = tdf.iloc[i + j]
                future_close = future.get("Close", entry_price)
                
                if direction == "LONG":
                    pnl_pct = (future_close - entry_price) / entry_price
                    if pnl_pct >= take_profit:
                        exit_price = future_close
                        exit_reason = "take_profit"
                        exit_day = j
                        break
                    elif pnl_pct <= -stop_loss:
                        exit_price = future_close
                        exit_reason = "stop_loss"
                        exit_day = j
                        break
                else:  # SHORT
                    pnl_pct = (entry_price - future_close) / entry_price
                    if pnl_pct >= take_profit:
                        exit_price = future_close
                        exit_reason = "take_profit"
                        exit_day = j
                        break
                    elif pnl_pct <= -stop_loss:
                        exit_price = future_close
                        exit_reason = "stop_loss"
                        exit_day = j
                        break
                
                exit_price = future_close
            
            if direction == "LONG":
                final_pnl = (exit_price - entry_price) / entry_price
            else:
                final_pnl = (entry_price - exit_price) / entry_price
            
            trades.append({
                "ticker": ticker,
                "direction": direction,
                "rsi": round(rsi, 1),
                "entry": round(entry_price, 2),
                "exit": round(exit_price, 2),
                "pnl_pct": round(final_pnl * 100, 2),
                "exit_reason": exit_reason,
                "days_held": exit_day,
                "vol_ratio": round(row.get("volume_ratio", 0), 2),
                "atr_pct": round(row.get("atr_pct", 0), 2),
            })
    
    return trades


def analyze_trades(trades, label=""):
    """Analyze trade results."""
    if not trades:
        print(f"  {label}: No trades")
        return {}
    
    df = pd.DataFrame(trades)
    wins = df[df["pnl_pct"] > 0]
    losses = df[df["pnl_pct"] <= 0]
    
    win_rate = len(wins) / len(df) * 100
    avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    total_pnl = df["pnl_pct"].sum()
    
    gross_profit = wins["pnl_pct"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl_pct"].sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    avg_days = df["days_held"].mean()
    
    # Exit reason breakdown
    tp = len(df[df["exit_reason"] == "take_profit"])
    sl = len(df[df["exit_reason"] == "stop_loss"])
    exp = len(df[df["exit_reason"] == "expired"])
    
    result = {
        "trades": len(df),
        "win_rate": round(win_rate, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl": round(total_pnl, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_days": round(avg_days, 1),
        "take_profit": tp,
        "stop_loss": sl,
        "expired": exp,
        "long": len(df[df["direction"] == "LONG"]),
        "short": len(df[df["direction"] == "SHORT"]),
    }
    
    print(f"\n📊 {label}")
    print(f"   Trades: {result['trades']} ({result['long']}L / {result['short']}S)")
    print(f"   Win Rate: {result['win_rate']}%")
    print(f"   Avg Win: +{result['avg_win']}% | Avg Loss: {result['avg_loss']}%")
    print(f"   Total P/L: {result['total_pnl']}%")
    print(f"   Profit Factor: {result['profit_factor']}")
    print(f"   Avg Hold: {result['avg_days']} days")
    print(f"   Exits: {tp} TP / {sl} SL / {exp} expired")
    
    return result


if __name__ == "__main__":
    print("🔄 Mean Reversion Backtest")
    print("=" * 60)
    
    # Test 1: No filters (raw RSI extreme)
    print("\n--- TEST 1: Raw RSI Extremes (no neural net filters) ---")
    trades_raw = run_reversion_backtest(vol_filter=False, volume_filter=False)
    r1 = analyze_trades(trades_raw, "Raw RSI <30/>70")
    
    # Test 2: Neural net volatility filter only
    print("\n--- TEST 2: RSI + Volatility Filter ---")
    trades_vol = run_reversion_backtest(vol_filter=True, volume_filter=False)
    r2 = analyze_trades(trades_vol, "RSI + Volatility")
    
    # Test 3: Neural net full filter (volatility + volume)
    print("\n--- TEST 3: RSI + Volatility + Volume (Neural Net Informed) ---")
    trades_full = run_reversion_backtest(vol_filter=True, volume_filter=True)
    r3 = analyze_trades(trades_full, "RSI + Vol + Volume (Neural)")
    
    # Test 4: Tighter RSI
    print("\n--- TEST 4: Tight RSI <25/>75 + Neural Filters ---")
    trades_tight = run_reversion_backtest(rsi_oversold=25, rsi_overbought=75, vol_filter=True, volume_filter=True)
    r4 = analyze_trades(trades_tight, "RSI <25/>75 + Neural")
    
    # Test 5: Inverted volume — LOW volume at extremes (quiet dips)
    print("\n--- TEST 5: RSI + Volatility + LOW Volume (inverted) ---")
    trades_low_vol = run_reversion_backtest(vol_filter=True, volume_filter=True, invert_volume=True)
    r5 = analyze_trades(trades_low_vol, "RSI + Vol + LOW Volume")
    
    # Test 6: Tight RSI + LOW volume
    print("\n--- TEST 6: Tight RSI <25/>75 + Vol + LOW Volume ---")
    trades_tight_low = run_reversion_backtest(rsi_oversold=25, rsi_overbought=75, vol_filter=True, volume_filter=True, invert_volume=True)
    r6 = analyze_trades(trades_tight_low, "Tight RSI + LOW Vol")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 HEAD-TO-HEAD")
    print(f"{'Strategy':<35} {'Trades':>7} {'WR':>7} {'PF':>7} {'P/L':>8}")
    print("-" * 65)
    for name, r in [("Raw RSI", r1), ("RSI + Volatility", r2), ("RSI + Vol + HIGH Volume", r3), ("Tight RSI + HIGH Vol", r4), ("RSI + Vol + LOW Volume", r5), ("Tight RSI + LOW Vol", r6)]:
        if r:
            print(f"{name:<35} {r['trades']:>7} {r['win_rate']:>6}% {r['profit_factor']:>7} {r['total_pnl']:>7}%")
    
    # Save
    results = {"raw": r1, "volatility": r2, "neural_full": r3, "tight_neural": r4, "low_volume": r5, "tight_low_vol": r6}
    with open(os.path.join(os.path.dirname(__file__), "reversion_backtest_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to reversion_backtest_results.json")
