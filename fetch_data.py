#!/usr/bin/env python3
"""
Neural Net Data Fetcher — Completely independent from Opscan/ThetaScan.
Pulls raw OHLCV data and computes technical indicators from scratch.
No factor scores, no signals DB, no scoring logic.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Full S&P 500 (~501 tickers)
TICKERS = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP",
    "APTV", "ARE", "ARES", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP",
    "AZO", "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BG", "BIIB",
    "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRO", "BSX", "BX",
    "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI",
    "CCL", "CDNS", "CDW", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI",
    "CIEN", "CINF", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC",
    "CNP", "COF", "COIN", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT",
    "CPT", "CRH", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTRA",
    "CTSH", "CTVA", "CVNA", "CVS", "CVX", "D", "DAL", "DASH", "DD", "DDOG",
    "DE", "DECK", "DELL", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR",
    "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM",
    "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EME",
    "EMR", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE", "ES", "ESS", "ETN",
    "ETR", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR", "F", "FANG",
    "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FICO", "FIS", "FISV", "FITB",
    "FIX", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE",
    "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOG",
    "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA",
    "HD", "HIG", "HII", "HLT", "HOLX", "HON", "HOOD", "HPE", "HPQ", "HRL",
    "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR", "IBM", "ICE", "IDXX",
    "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IQV", "IR", "IRM",
    "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ",
    "JPM", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI",
    "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX", "LII", "LIN",
    "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB",
    "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ",
    "MDT", "MET", "META", "MGM", "MKC", "MLM", "MMM", "MNST", "MO", "MOH",
    "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRSH", "MS", "MSCI", "MSFT", "MSI",
    "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX",
    "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA",
    "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL",
    "ORLY", "OTIS", "OXY", "PANW", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP",
    "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM",
    "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX",
    "PTC", "PWR", "PYPL", "QCOM", "RCL", "REG", "REGN", "RF", "RJF", "RL",
    "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX",
    "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG",
    "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS",
    "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER",
    "TFC", "TGT", "TJX", "TKO", "TMO", "TMUS", "TPL", "TPR", "TRGP", "TRMB",
    "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTD", "TTWO", "TXN", "TXT",
    "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI",
    "USB", "V", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST",
    "VTR", "VTRS", "VZ", "WAB", "WAT", "WBD", "WDAY", "WDC", "WEC", "WELL",
    "WFC", "WM", "WMB", "WMT", "WRB", "WSM", "WST", "WTW", "WY", "WYNN",
    "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS",
]


def compute_indicators(df):
    """Compute all technical indicators from raw OHLCV. No external data needed."""
    if len(df) < 50:
        return df

    # --- Moving Averages ---
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # --- MACD ---
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- RSI (14-period) ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands ---
    bb_sma = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_upper"] = bb_sma + 2 * bb_std
    df["bb_lower"] = bb_sma - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_sma
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # --- ATR (14-period) ---
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / df["Close"] * 100

    # --- ADX (14-period) ---
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.rolling(14).mean()
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # --- Stochastic Oscillator ---
    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()
    df["stoch_k"] = 100 * (df["Close"] - low_14) / (high_14 - low_14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- Volume indicators ---
    df["volume_sma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_sma_20"].replace(0, np.nan)

    # --- OBV (On-Balance Volume) ---
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["obv"] = obv
    df["obv_sma"] = obv.rolling(20).mean()

    # --- Price momentum ---
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)
    df["return_20d"] = df["Close"].pct_change(20)

    # --- Volatility ---
    df["volatility_10d"] = df["return_1d"].rolling(10).std() * np.sqrt(252)
    df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)

    # --- VWAP (daily approximation) ---
    df["vwap"] = (df["Close"] * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum().replace(0, np.nan)

    # --- RSI Divergence (using rsi-divergence-detector library) ---
    try:
        from rsi_divergence import find_divergences, wilder_rsi
        rsi_series = pd.Series(wilder_rsi(df["Close"], period=14), index=df.index, dtype=float)
        divs = find_divergences(df["Close"].astype(float), rsi_series, include_hidden=True)

        # Initialize divergence columns
        df["rsi_bull_divergence"] = 0
        df["rsi_bear_divergence"] = 0
        df["rsi_hidden_bull"] = 0
        df["rsi_hidden_bear"] = 0

        for _, d in divs.iterrows():
            idx = int(d["p2_idx"])  # Signal at the second pivot
            if idx < len(df):
                if d["kind"] == "regular_bullish":
                    df.iloc[idx, df.columns.get_loc("rsi_bull_divergence")] = 1
                elif d["kind"] == "regular_bearish":
                    df.iloc[idx, df.columns.get_loc("rsi_bear_divergence")] = 1
                elif d["kind"] == "hidden_bullish":
                    df.iloc[idx, df.columns.get_loc("rsi_hidden_bull")] = 1
                elif d["kind"] == "hidden_bearish":
                    df.iloc[idx, df.columns.get_loc("rsi_hidden_bear")] = 1

        # Combined: +1 bullish (regular+hidden), -1 bearish (regular+hidden), 0 none
        df["rsi_divergence"] = (df["rsi_bull_divergence"] + df["rsi_hidden_bull"]) - \
                                (df["rsi_bear_divergence"] + df["rsi_hidden_bear"])
    except ImportError:
        # Fallback if library not installed
        df["rsi_bull_divergence"] = 0
        df["rsi_bear_divergence"] = 0
        df["rsi_hidden_bull"] = 0
        df["rsi_hidden_bear"] = 0
        df["rsi_divergence"] = 0
    except Exception as e:
        print(f"  RSI divergence detection failed for ticker: {e}")
        df["rsi_bull_divergence"] = 0
        df["rsi_bear_divergence"] = 0
        df["rsi_hidden_bull"] = 0
        df["rsi_hidden_bear"] = 0
        df["rsi_divergence"] = 0

    # --- Trend direction features ---
    df["above_sma20"] = (df["Close"] > df["sma_20"]).astype(int)
    df["above_sma50"] = (df["Close"] > df["sma_50"]).astype(int)
    df["sma_cross"] = (df["sma_5"] > df["sma_20"]).astype(int)

    # --- Range position ---
    daily_range = df["High"] - df["Low"]
    df["daily_range_pct"] = daily_range / df["Close"] * 100
    df["close_in_range"] = (df["Close"] - df["Low"]) / daily_range.replace(0, np.nan)

    return df


def create_labels(df, forward_hours=4):
    """
    Create buy/sell labels based on future price movement.
    Using daily data: forward_hours=4 ≈ 1 day, forward_hours=8 ≈ 2 days.
    Label: 1 = price went up by 0.5%+, 0 = neutral/down, -1 = down 0.5%+
    """
    forward_days = max(1, forward_hours // 4)
    future_return = df["Close"].shift(-forward_days) / df["Close"] - 1

    # Three classes: buy (>0.5%), sell (<-0.5%), neutral
    labels = pd.Series(0, index=df.index)  # neutral
    labels[future_return > 0.005] = 1   # buy
    labels[future_return < -0.005] = -1  # sell

    df["label"] = labels
    df["future_return"] = future_return
    return df


def fetch_ticker(ticker, period="5y", interval="1d"):
    """Fetch OHLCV for a single ticker."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data is None or len(data) < 100:
            return None
        data = data.reset_index()
        # Flatten multi-level columns if present
        if hasattr(data.columns, 'levels'):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
        # Normalize date column name (yfinance uses "Datetime" for intraday)
        if "Datetime" in data.columns and "Date" not in data.columns:
            data = data.rename(columns={"Datetime": "Date"})
        if "Date" not in data.columns:
            # Date might still be in index after reset
            for col in data.columns:
                if "date" in str(col).lower() or "time" in str(col).lower():
                    data = data.rename(columns={col: "Date"})
                    break
        data["ticker"] = ticker
        return data
    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def fetch_all(tickers=None, period="5y"):
    """Fetch and process all tickers."""
    if tickers is None:
        tickers = TICKERS

    all_data = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        print(f"  [{i+1}/{total}] Fetching {ticker}...")
        df = fetch_ticker(ticker, period=period)
        if df is not None and len(df) > 50:
            df = compute_indicators(df)
            df = create_labels(df)
            all_data.append(df)
        time.sleep(0.2)  # Rate limit

    if not all_data:
        print("ERROR: No data fetched!")
        return None

    combined = pd.concat(all_data, ignore_index=True)

    # Drop rows with NaN in indicators (first ~50 rows per ticker)
    indicator_cols = [c for c in combined.columns if c not in ["Date", "ticker", "label", "future_return"]]
    combined = combined.dropna(subset=["label"])
    combined = combined.dropna(subset=indicator_cols, how="any")

    # Save
    out_path = os.path.join(DATA_DIR, "training_data.csv")
    combined.to_csv(out_path, index=False)

    # Save metadata
    date_range = ["unknown", "unknown"]
    if "Date" in combined.columns:
        date_range = [str(combined["Date"].min()), str(combined["Date"].max())]
    meta = {
        "tickers": tickers,
        "total_rows": len(combined),
        "date_range": date_range,
        "features": indicator_cols,
        "label_distribution": combined["label"].value_counts().to_dict(),
        "fetched_at": datetime.now().isoformat(),
        "period": period
    }
    with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n✅ Saved {len(combined)} rows across {len(tickers)} tickers")
    print(f"   Labels: {meta['label_distribution']}")
    print(f"   Features: {len(indicator_cols)}")
    return combined


if __name__ == "__main__":
    print("🧠 Neural Net Data Fetcher — Independent System")
    print("=" * 50)
    fetch_all()
