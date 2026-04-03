#!/usr/bin/env python3
"""
Bimodal Volume Profile Detector
================================
Analyzes NY sessions for bimodal volume distribution.
"""

import argparse
from datetime import datetime, timedelta
import numpy as np

try:
    import yfinance as yf
    from scipy import signal
    from scipy.stats import gaussian_kde
except ImportError as e:
    print(f"Install: pip install yfinance scipy")
    exit(1)


def detect_bimodal(prices, volumes, num_bins=50):
    """
    Detect if volume profile is bimodal.
    Returns: (is_bimodal, peak_prices, profile_data)
    """
    if len(prices) < 10 or sum(volumes) == 0:
        return False, [], {}
    
    # Create price bins
    price_min, price_max = min(prices), max(prices)
    if price_max == price_min:
        return False, [], {}
    
    bins = np.linspace(price_min, price_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Sum volume in each bin
    volume_profile = np.zeros(num_bins)
    for p, v in zip(prices, volumes):
        bin_idx = min(int((p - price_min) / (price_max - price_min) * num_bins), num_bins - 1)
        volume_profile[bin_idx] += v
    
    # Smooth the profile
    if len(volume_profile) > 5:
        window = min(5, len(volume_profile) // 2)
        if window > 0:
            kernel = np.ones(window) / window
            volume_profile_smooth = np.convolve(volume_profile, kernel, mode='same')
        else:
            volume_profile_smooth = volume_profile
    else:
        volume_profile_smooth = volume_profile
    
    # Find peaks
    peaks, properties = signal.find_peaks(
        volume_profile_smooth, 
        height=np.max(volume_profile_smooth) * 0.3,  # At least 30% of max
        distance=num_bins // 5,  # Peaks must be separated
        prominence=np.max(volume_profile_smooth) * 0.15  # Prominence threshold
    )
    
    is_bimodal = len(peaks) >= 2
    peak_prices = [bin_centers[p] for p in peaks[:2]] if len(peaks) >= 2 else []
    
    return is_bimodal, peak_prices, {
        'bins': bin_centers.tolist(),
        'volume': volume_profile.tolist(),
        'peaks': peaks.tolist(),
    }


def analyze_sessions(symbol: str, days: int = 60):
    """Analyze NY sessions for bimodal volume profiles."""
    
    print(f"Fetching {symbol} intraday data ({days} days)...")
    
    # yfinance: max 60 days for 5m data
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{min(days, 60)}d", interval="5m")
    
    if df.empty:
        print("No data found")
        return
    
    # Filter to NY session (9:30 AM - 4:00 PM ET)
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['time_decimal'] = df['hour'] + df['minute'] / 60
    
    # NY session: 9:30 (9.5) to 16:00 (16.0)
    df_ny = df[(df['time_decimal'] >= 9.5) & (df['time_decimal'] < 16.0)]
    
    # Group by date
    df_ny = df_ny.copy()
    df_ny['date'] = df_ny.index.date
    
    results = {
        'total_sessions': 0,
        'bimodal_sessions': 0,
        'bimodal_dates': [],
        'unimodal_dates': [],
        'spreads': [],
        'spread_pcts': [],
    }
    
    for date, group in df_ny.groupby('date'):
        if len(group) < 20:  # Need enough bars
            continue
            
        results['total_sessions'] += 1
        
        # Use typical price * volume for VWAP-style weighting
        typical_price = (group['High'] + group['Low'] + group['Close']) / 3
        mid_price = typical_price.mean()
        volumes = group['Volume'].values
        prices = typical_price.values
        
        is_bimodal, peak_prices, _ = detect_bimodal(prices, volumes)
        
        if is_bimodal:
            results['bimodal_sessions'] += 1
            
            # Calculate spread between peaks
            if len(peak_prices) >= 2:
                spread = abs(peak_prices[1] - peak_prices[0])
                spread_pct = spread / mid_price * 100
                results['spreads'].append(spread)
                results['spread_pcts'].append(spread_pct)
            
            results['bimodal_dates'].append({
                'date': str(date),
                'peaks': [round(p, 2) for p in peak_prices],
                'spread': round(spread, 2) if len(peak_prices) >= 2 else 0,
                'spread_pct': round(spread_pct, 3) if len(peak_prices) >= 2 else 0,
            })
        else:
            results['unimodal_dates'].append(str(date))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Bimodal Volume Profile Detector')
    parser.add_argument('--symbol', type=str, default='SPY', help='Symbol')
    parser.add_argument('--days', type=int, default=60, help='Days to analyze')
    args = parser.parse_args()
    
    print("=" * 60)
    print("BIMODAL VOLUME PROFILE ANALYSIS")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Session: NY (9:30 AM - 4:00 PM ET)")
    print("=" * 60)
    
    results = analyze_sessions(args.symbol, args.days)
    
    if not results:
        return
    
    bimodal_pct = (results['bimodal_sessions'] / results['total_sessions'] * 100) if results['total_sessions'] > 0 else 0
    
    print(f"\nTotal NY Sessions: {results['total_sessions']}")
    print(f"Bimodal Sessions: {results['bimodal_sessions']}")
    print(f"Bimodal Rate: {bimodal_pct:.1f}%")
    print()
    
    # Peak-to-peak spread statistics
    if results['spreads']:
        spreads = np.array(results['spreads'])
        spread_pcts = np.array(results['spread_pcts'])
        
        print("\nPeak-to-Peak Distance:")
        print(f"  Avg spread:    ${np.mean(spreads):.2f} ({np.mean(spread_pcts):.2f}%)")
        print(f"  Median spread: ${np.median(spreads):.2f} ({np.median(spread_pcts):.2f}%)")
        print(f"  Min spread:    ${np.min(spreads):.2f} ({np.min(spread_pcts):.2f}%)")
        print(f"  Max spread:    ${np.max(spreads):.2f} ({np.max(spread_pcts):.2f}%)")
        
        print("\nSpread Distribution:")
        print(f"  < 0.25%:     {sum(1 for x in spread_pcts if x < 0.25)} sessions")
        print(f"  0.25-0.50%:  {sum(1 for x in spread_pcts if 0.25 <= x < 0.50)} sessions")
        print(f"  0.50-0.75%:  {sum(1 for x in spread_pcts if 0.50 <= x < 0.75)} sessions")
        print(f"  0.75-1.00%:  {sum(1 for x in spread_pcts if 0.75 <= x < 1.00)} sessions")
        print(f"  > 1.00%:     {sum(1 for x in spread_pcts if x >= 1.00)} sessions")
    
    if results['bimodal_dates']:
        print("\nRecent Bimodal Days:")
        for item in results['bimodal_dates'][-10:]:
            print(f"  {item['date']}: peaks at {item['peaks']} (spread: ${item.get('spread', 0):.2f} / {item.get('spread_pct', 0):.2f}%)")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
