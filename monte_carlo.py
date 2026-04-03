#!/usr/bin/env python3
"""
Monte Carlo Simulation for RSI + Stochastic Strategy
=====================================================
Runs Monte Carlo simulation on backtest results to estimate:
- Expected return distribution
- Risk of ruin
- Confidence intervals

Usage:
    python monte_carlo.py --symbol SPY --days 1460 --simulations 10000
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

# Import the backtest function
from backtest_rsi_stoch_alignment import fetch_data, backtest_alignment


def monte_carlo_simulation(returns: list, 
                           num_simulations: int = 10000,
                           num_trades: int = 100,
                           initial_capital: float = 10000) -> dict:
    """
    Run Monte Carlo simulation on historical trade returns.
    
    Args:
        returns: List of historical trade returns (as percentages)
        num_simulations: Number of simulation runs
        num_trades: Number of trades per simulation
        initial_capital: Starting capital
    
    Returns:
        Dict with simulation statistics
    """
    if not returns:
        return {'error': 'No returns data'}
    
    returns = np.array(returns) / 100  # Convert to decimal
    
    final_capitals = []
    max_drawdowns = []
    all_sim_returns = []  # Track individual simulation returns for Sortino
    ruin_count = 0
    ruin_threshold = 0.5  # 50% drawdown = ruin
    
    for _ in range(num_simulations):
        capital = initial_capital
        peak = capital
        max_dd = 0
        
        # Randomly sample trades with replacement
        sim_returns = np.random.choice(returns, size=num_trades, replace=True)
        
        for ret in sim_returns:
            capital *= (1 + ret)
            
            # Track peak and drawdown
            if capital > peak:
                peak = capital
            
            dd = (peak - capital) / peak
            if dd > max_dd:
                max_dd = dd
            
            # Check for ruin
            if dd >= ruin_threshold:
                ruin_count += 1
                break
        
        final_capitals.append(capital)
        max_drawdowns.append(max_dd * 100)
        all_sim_returns.append((capital - initial_capital) / initial_capital)
    
    final_capitals = np.array(final_capitals)
    max_drawdowns = np.array(max_drawdowns)
    all_sim_returns = np.array(all_sim_returns)
    
    # Calculate statistics
    returns_pct = ((final_capitals - initial_capital) / initial_capital) * 100
    
    # Sortino Ratio: only penalize downside volatility
    # Downside deviation = std of returns below target (0)
    downside_returns = all_sim_returns[all_sim_returns < 0]
    if len(downside_returns) > 0:
        downside_std = np.std(downside_returns)
        sortino = float(np.mean(all_sim_returns) / downside_std) if downside_std > 0 else float('inf')
    else:
        sortino = float('inf')  # No downside = infinite Sortino
    
    # Drawdown confidence intervals
    dd_percentile_50 = float(np.percentile(max_drawdowns, 50))
    dd_percentile_75 = float(np.percentile(max_drawdowns, 75))
    dd_percentile_90 = float(np.percentile(max_drawdowns, 90))
    dd_percentile_95 = float(np.percentile(max_drawdowns, 95))
    dd_percentile_99 = float(np.percentile(max_drawdowns, 99))
    
    # Strategy "broken" threshold: 99th percentile DD
    # If you exceed this in live trading, strategy may be dead
    strategy_broken_dd = dd_percentile_99
    
    return {
        'simulations': num_simulations,
        'trades_per_sim': num_trades,
        'initial_capital': initial_capital,
        'mean_return': float(np.mean(returns_pct)),
        'median_return': float(np.median(returns_pct)),
        'std_return': float(np.std(returns_pct)),
        'min_return': float(np.min(returns_pct)),
        'max_return': float(np.max(returns_pct)),
        'percentile_5': float(np.percentile(returns_pct, 5)),
        'percentile_25': float(np.percentile(returns_pct, 25)),
        'percentile_75': float(np.percentile(returns_pct, 75)),
        'percentile_95': float(np.percentile(returns_pct, 95)),
        'prob_profit': float(np.mean(returns_pct > 0) * 100),
        'prob_double': float(np.mean(returns_pct >= 100) * 100),
        'risk_of_ruin': float(ruin_count / num_simulations * 100),
        'mean_max_drawdown': float(np.mean(max_drawdowns)),
        'worst_drawdown': float(np.max(max_drawdowns)),
        'sharpe_estimate': float(np.mean(returns_pct) / np.std(returns_pct)) if np.std(returns_pct) > 0 else 0,
        'sortino_ratio': sortino,
        'dd_percentile_50': dd_percentile_50,
        'dd_percentile_75': dd_percentile_75,
        'dd_percentile_90': dd_percentile_90,
        'dd_percentile_95': dd_percentile_95,
        'dd_percentile_99': dd_percentile_99,
        'strategy_broken_threshold': strategy_broken_dd,
    }


def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Simulation')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol')
    parser.add_argument('--days', type=int, default=1460, help='Days of data')
    parser.add_argument('--simulations', type=int, default=10000, help='Number of simulations')
    parser.add_argument('--trades', type=int, default=100, help='Trades per simulation')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--rsi-high', type=float, default=70)
    parser.add_argument('--rsi-low', type=float, default=30)
    parser.add_argument('--stoch-high', type=float, default=70)
    parser.add_argument('--stoch-low', type=float, default=30)
    parser.add_argument('--hold', type=int, default=5)
    parser.add_argument('--stop', type=float, default=2.0)
    parser.add_argument('--target', type=float, default=3.0)
    parser.add_argument('--output', type=str, default='training/results')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MONTE CARLO SIMULATION")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Historical Data: {args.days} days")
    print(f"Simulations: {args.simulations:,}")
    print(f"Trades per sim: {args.trades}")
    print(f"Initial Capital: ${args.capital:,.0f}")
    print("=" * 60)
    
    # Fetch data and run backtest
    print("\nRunning backtest to get trade returns...")
    data = fetch_data(args.symbol, args.days)
    
    backtest_results = backtest_alignment(
        data,
        rsi_high=args.rsi_high,
        rsi_low=args.rsi_low,
        stoch_high=args.stoch_high,
        stoch_low=args.stoch_low,
        hold_bars=args.hold,
        stop_pct=args.stop,
        target_pct=args.target,
    )
    
    if backtest_results['total_trades'] < 5:
        print(f"\nError: Only {backtest_results['total_trades']} trades found. Need at least 5 for Monte Carlo.")
        return
    
    # Get trade returns
    returns = [t['pnl_pct'] for t in backtest_results['trades']]
    
    print(f"Found {len(returns)} historical trades")
    print(f"Historical Win Rate: {backtest_results['win_rate']:.1f}%")
    print(f"Historical Avg Return: {np.mean(returns):.2f}%")
    
    # Run Monte Carlo
    print(f"\nRunning {args.simulations:,} simulations...")
    mc_results = monte_carlo_simulation(
        returns,
        num_simulations=args.simulations,
        num_trades=args.trades,
        initial_capital=args.capital,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("MONTE CARLO RESULTS")
    print("=" * 60)
    print(f"After {args.trades} trades starting with ${args.capital:,.0f}:")
    print()
    print(f"Expected Return: {mc_results['mean_return']:.1f}% (median: {mc_results['median_return']:.1f}%)")
    print(f"Std Dev: {mc_results['std_return']:.1f}%")
    print()
    print("Return Distribution:")
    print(f"  5th percentile:  {mc_results['percentile_5']:.1f}%")
    print(f"  25th percentile: {mc_results['percentile_25']:.1f}%")
    print(f"  75th percentile: {mc_results['percentile_75']:.1f}%")
    print(f"  95th percentile: {mc_results['percentile_95']:.1f}%")
    print()
    print(f"Best case:  {mc_results['max_return']:.1f}%")
    print(f"Worst case: {mc_results['min_return']:.1f}%")
    print()
    print(f"Probability of profit: {mc_results['prob_profit']:.1f}%")
    print(f"Probability of doubling: {mc_results['prob_double']:.1f}%")
    print(f"Risk of ruin (50% DD): {mc_results['risk_of_ruin']:.1f}%")
    print()
    print(f"Mean Max Drawdown: {mc_results['mean_max_drawdown']:.1f}%")
    print(f"Worst Drawdown: {mc_results['worst_drawdown']:.1f}%")
    print()
    print(f"Sharpe Ratio: {mc_results['sharpe_estimate']:.2f}")
    print(f"Sortino Ratio: {mc_results['sortino_ratio']:.2f}")
    print()
    print("Drawdown Confidence Intervals:")
    print(f"  50% of runs stay under: {mc_results['dd_percentile_50']:.1f}% DD")
    print(f"  75% of runs stay under: {mc_results['dd_percentile_75']:.1f}% DD")
    print(f"  90% of runs stay under: {mc_results['dd_percentile_90']:.1f}% DD")
    print(f"  95% of runs stay under: {mc_results['dd_percentile_95']:.1f}% DD")
    print(f"  99% of runs stay under: {mc_results['dd_percentile_99']:.1f}% DD")
    print()
    print(f"*** STRATEGY BROKEN THRESHOLD: {mc_results['strategy_broken_threshold']:.1f}% DD ***")
    print(f"    (If you hit this DD live, strategy may be dead)")
    print("=" * 60)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / f"{args.symbol}_monte_carlo.json"
    
    save_data = {
        'symbol': args.symbol,
        'params': {
            'days': args.days,
            'simulations': args.simulations,
            'trades_per_sim': args.trades,
            'initial_capital': args.capital,
            'rsi_high': args.rsi_high,
            'rsi_low': args.rsi_low,
            'stoch_high': args.stoch_high,
            'stoch_low': args.stoch_low,
            'hold': args.hold,
            'stop': args.stop,
            'target': args.target,
        },
        'backtest': {
            'total_trades': backtest_results['total_trades'],
            'win_rate': backtest_results['win_rate'],
            'sharpe': backtest_results['sharpe'],
            'profit_factor': backtest_results['profit_factor'],
        },
        'monte_carlo': mc_results,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
