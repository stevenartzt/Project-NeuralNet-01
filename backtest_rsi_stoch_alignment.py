#!/usr/bin/env python3
"""
RSI + Stochastic Alignment Backtest
====================================
Tests the specific strategy:
- SHORT: RSI > 75 AND Stochastic %K crossing below %D
- LONG: RSI < 25 AND Stochastic %K crossing above %D

Usage:
    python backtest_rsi_stoch_alignment.py --symbol SPY --days 730
    python backtest_rsi_stoch_alignment.py --symbol NQ=F --days 365
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    import yfinance as yf
except ImportError:
    print("Install yfinance: pip install yfinance")
    exit(1)


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI indicator."""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi = np.zeros(len(prices))
    
    # Initial averages
    if len(gains) >= period:
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))
        else:
            rsi[period] = 100
        
        # Smoothed averages
        for i in range(period + 1, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
    
    return rsi


def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                          k_period: int = 14, d_period: int = 3) -> tuple:
    """Calculate Stochastic %K and %D."""
    stoch_k = np.zeros(len(close))
    stoch_d = np.zeros(len(close))
    
    for i in range(k_period - 1, len(close)):
        highest_high = np.max(high[i - k_period + 1:i + 1])
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        
        if highest_high != lowest_low:
            stoch_k[i] = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
        else:
            stoch_k[i] = 50
    
    # %D is SMA of %K
    for i in range(k_period + d_period - 2, len(close)):
        stoch_d[i] = np.mean(stoch_k[i - d_period + 1:i + 1])
    
    return stoch_k, stoch_d


def fetch_data(symbol: str, days: int = 365) -> dict:
    """Fetch OHLCV data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Fetching {symbol} data ({days} days)...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    
    return {
        'open': df['Open'].values,
        'high': df['High'].values,
        'low': df['Low'].values,
        'close': df['Close'].values,
        'volume': df['Volume'].values,
        'dates': df.index.tolist(),
    }


def backtest_alignment(data: dict, 
                       rsi_high: float = 75,
                       rsi_low: float = 25,
                       stoch_high: float = 80,
                       stoch_low: float = 20,
                       hold_bars: int = 5,
                       stop_pct: float = 2.0,
                       target_pct: float = 3.0) -> dict:
    """
    Backtest RSI + Stochastic alignment strategy.
    
    Entry rules:
    - SHORT: RSI > rsi_high AND Stoch %K crosses below %D (both > stoch_high)
    - LONG: RSI < rsi_low AND Stoch %K crosses above %D (both < stoch_low)
    
    Exit rules:
    - Hold for N bars OR hit stop/target
    """
    close = data['close']
    high = data['high']
    low = data['low']
    
    # Calculate indicators
    rsi = calculate_rsi(close, 14)
    stoch_k, stoch_d = calculate_stochastic(high, low, close, 14, 3)
    
    trades = []
    signals = []
    
    start_idx = 20  # Ensure indicators are valid
    
    i = start_idx
    while i < len(close) - hold_bars:
        signal = None
        
        # Check for SHORT signal
        # RSI > 75 AND Stoch %K was above %D, now crossing below
        if (rsi[i] > rsi_high and 
            stoch_k[i] > stoch_high and stoch_d[i] > stoch_high and
            stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1]):
            signal = 'short'
        
        # Check for LONG signal
        # RSI < 25 AND Stoch %K was below %D, now crossing above
        elif (rsi[i] < rsi_low and 
              stoch_k[i] < stoch_low and stoch_d[i] < stoch_low and
              stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1]):
            signal = 'long'
        
        if signal:
            entry_price = close[i]
            entry_date = data['dates'][i] if i < len(data['dates']) else i
            
            signals.append({
                'date': str(entry_date),
                'signal': signal,
                'entry': entry_price,
                'rsi': rsi[i],
                'stoch_k': stoch_k[i],
                'stoch_d': stoch_d[i],
            })
            
            # Simulate trade
            exit_price = None
            exit_reason = None
            
            for j in range(1, hold_bars + 1):
                if i + j >= len(close):
                    break
                
                if signal == 'long':
                    # Check stop loss
                    if low[i + j] <= entry_price * (1 - stop_pct / 100):
                        exit_price = entry_price * (1 - stop_pct / 100)
                        exit_reason = 'stop'
                        break
                    # Check target
                    if high[i + j] >= entry_price * (1 + target_pct / 100):
                        exit_price = entry_price * (1 + target_pct / 100)
                        exit_reason = 'target'
                        break
                else:  # short
                    # Check stop loss (price goes up)
                    if high[i + j] >= entry_price * (1 + stop_pct / 100):
                        exit_price = entry_price * (1 + stop_pct / 100)
                        exit_reason = 'stop'
                        break
                    # Check target (price goes down)
                    if low[i + j] <= entry_price * (1 - target_pct / 100):
                        exit_price = entry_price * (1 - target_pct / 100)
                        exit_reason = 'target'
                        break
            
            # Exit at end of hold period if no stop/target hit
            if exit_price is None:
                exit_price = close[min(i + hold_bars, len(close) - 1)]
                exit_reason = 'time'
            
            # Calculate P/L
            if signal == 'long':
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            
            trades.append({
                'signal': signal,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'pnl_pct': pnl_pct,
                'rsi': rsi[i],
                'stoch_k': stoch_k[i],
                'stoch_d': stoch_d[i],
            })
            
            # Skip ahead past this trade
            i += hold_bars
        else:
            i += 1
    
    # Calculate stats
    if not trades:
        return {
            'total_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'win_rate': 0,
            'sharpe': 0,
            'total_return': 0,
            'avg_return': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'target_hits': 0,
            'stop_hits': 0,
            'time_exits': 0,
            'signals': signals,
            'trades': trades,
        }
    
    returns = np.array([t['pnl_pct'] for t in trades])
    wins = sum(1 for t in trades if t['pnl_pct'] > 0)
    
    # Sharpe ratio (annualized)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 / hold_bars)
    else:
        sharpe = 0
    
    # Profit factor
    gross_profit = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    gross_loss = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max()
    
    return {
        'total_trades': len(trades),
        'long_trades': sum(1 for t in trades if t['signal'] == 'long'),
        'short_trades': sum(1 for t in trades if t['signal'] == 'short'),
        'win_rate': wins / len(trades) * 100,
        'sharpe': sharpe,
        'total_return': cumulative[-1],
        'avg_return': returns.mean(),
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'target_hits': sum(1 for t in trades if t['exit_reason'] == 'target'),
        'stop_hits': sum(1 for t in trades if t['exit_reason'] == 'stop'),
        'time_exits': sum(1 for t in trades if t['exit_reason'] == 'time'),
        'avg_win': np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]) if wins > 0 else 0,
        'avg_loss': np.mean([t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]) if wins < len(trades) else 0,
        'signals': signals,
        'trades': trades,
    }


def main():
    parser = argparse.ArgumentParser(description='Backtest RSI + Stochastic Alignment')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol')
    parser.add_argument('--days', type=int, default=730, help='Days of data')
    parser.add_argument('--rsi-high', type=float, default=75, help='RSI overbought threshold')
    parser.add_argument('--rsi-low', type=float, default=25, help='RSI oversold threshold')
    parser.add_argument('--stoch-high', type=float, default=80, help='Stochastic overbought')
    parser.add_argument('--stoch-low', type=float, default=20, help='Stochastic oversold')
    parser.add_argument('--hold', type=int, default=5, help='Hold period (bars)')
    parser.add_argument('--stop', type=float, default=2.0, help='Stop loss %')
    parser.add_argument('--target', type=float, default=3.0, help='Target %')
    parser.add_argument('--output', type=str, default='training/results', help='Output dir')
    args = parser.parse_args()
    
    print("=" * 60)
    print("RSI + STOCHASTIC ALIGNMENT BACKTEST")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Days: {args.days}")
    print(f"RSI Thresholds: < {args.rsi_low} (long), > {args.rsi_high} (short)")
    print(f"Stochastic: < {args.stoch_low} (long), > {args.stoch_high} (short)")
    print(f"Hold: {args.hold} bars, Stop: {args.stop}%, Target: {args.target}%")
    print("=" * 60)
    
    # Fetch data
    data = fetch_data(args.symbol, args.days)
    print(f"Loaded {len(data['close'])} bars")
    
    # Run backtest
    results = backtest_alignment(
        data,
        rsi_high=args.rsi_high,
        rsi_low=args.rsi_low,
        stoch_high=args.stoch_high,
        stoch_low=args.stoch_low,
        hold_bars=args.hold,
        stop_pct=args.stop,
        target_pct=args.target,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total Trades: {results['total_trades']}")
    print(f"  Long: {results['long_trades']}")
    print(f"  Short: {results['short_trades']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Sharpe Ratio: {results['sharpe']:.2f}")
    print(f"Total Return: {results['total_return']:.1f}%")
    print(f"Avg Return/Trade: {results['avg_return']:.2f}%")
    print(f"Max Drawdown: {results['max_drawdown']:.1f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Avg Win: {results.get('avg_win', 0):.2f}%")
    print(f"Avg Loss: {results.get('avg_loss', 0):.2f}%")
    print(f"Exit Breakdown:")
    print(f"  Target hits: {results['target_hits']}")
    print(f"  Stop hits: {results['stop_hits']}")
    print(f"  Time exits: {results['time_exits']}")
    print("=" * 60)
    
    # Show recent signals
    if results['signals']:
        print("\nRecent Signals:")
        for sig in results['signals'][-5:]:
            print(f"  {sig['date']}: {sig['signal'].upper()} @ {sig['entry']:.2f} "
                  f"(RSI={sig['rsi']:.1f}, K={sig['stoch_k']:.1f}, D={sig['stoch_d']:.1f})")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{args.symbol}_alignment_backtest.json"
    
    # Remove non-serializable items for JSON
    save_results = {k: v for k, v in results.items() if k not in ['signals', 'trades']}
    save_results['params'] = {
        'symbol': args.symbol,
        'days': args.days,
        'rsi_high': args.rsi_high,
        'rsi_low': args.rsi_low,
        'stoch_high': args.stoch_high,
        'stoch_low': args.stoch_low,
        'hold_bars': args.hold,
        'stop_pct': args.stop,
        'target_pct': args.target,
    }
    save_results['timestamp'] = datetime.now().isoformat()
    
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == '__main__':
    main()
