#!/usr/bin/env python3
"""
Multi-Horizon Classifier — Testing momentum persistence hypothesis.
Hypothesis: Longer prediction horizons should show stronger signal because
momentum persists over days/weeks (documented in academic literature).

Tests: 1-day, 3-day, 5-day, 10-day, 20-day prediction horizons.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "horizon_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Features to use (no lookahead bias - all computed from past data)
FEATURE_COLS = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_position",
    "atr_pct", "adx", "plus_di", "minus_di",
    "stoch_k", "stoch_d",
    "volume_ratio",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_10d", "volatility_20d",
    "above_sma20", "above_sma50", "sma_cross",
    "daily_range_pct", "close_in_range",
]


class HorizonNet(nn.Module):
    """Binary classifier for multi-horizon prediction."""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def load_data_for_horizon(horizon_days, threshold_pct=0.5):
    """
    Load data and create labels for a specific prediction horizon.
    
    Args:
        horizon_days: How many days ahead to predict (1, 3, 5, 10, 20)
        threshold_pct: Minimum move to classify as UP/DOWN (0.5 = 0.5%)
    
    Returns:
        X, y, features
    """
    path = os.path.join(DATA_DIR, "training_data.csv")
    df = pd.read_csv(path)
    
    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Group by ticker to compute forward returns correctly
    results = []
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].copy()
        ticker_df = ticker_df.sort_values("Date")
        
        # Compute forward return for this horizon
        ticker_df[f"future_return_{horizon_days}d"] = (
            ticker_df["Close"].shift(-horizon_days) / ticker_df["Close"] - 1
        )
        results.append(ticker_df)
    
    df = pd.concat(results, ignore_index=True)
    
    # Create binary labels
    return_col = f"future_return_{horizon_days}d"
    df = df.dropna(subset=[return_col])
    
    threshold = threshold_pct / 100
    up = df[return_col] > threshold
    down = df[return_col] < -threshold
    
    # Keep only clear signals
    df = df[up | down].copy()
    df["binary_label"] = (df[return_col] > threshold).astype(int)
    
    # Extract features
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df["binary_label"].values
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, available, len(df)


def train_horizon(horizon_days, epochs=80, batch_size=128, lr=0.0003, threshold_pct=0.5):
    """Train model for a specific horizon."""
    
    X, y, features, n_samples = load_data_for_horizon(horizon_days, threshold_pct)
    
    up_pct = y.mean()
    down_pct = 1 - up_pct
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Model
    model = HorizonNet(input_size=len(features)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    best_val_acc = 0
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t).squeeze()
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)
        
        val_acc = accuracy_score(y_test, val_preds)
        val_auc = roc_auc_score(y_test, val_probs)
        
        scheduler.step(1 - val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 20:
            break
    
    return {
        "horizon_days": horizon_days,
        "samples": n_samples,
        "up_pct": f"{up_pct:.1%}",
        "accuracy": best_val_acc,
        "auc": best_val_auc,
        "threshold_pct": threshold_pct
    }


def run_horizon_sweep():
    """Test all horizons and compare."""
    print("=" * 70)
    print("MULTI-HORIZON CLASSIFIER — Testing Momentum Persistence")
    print("=" * 70)
    print()
    print("Hypothesis: Longer horizons should show stronger predictive signal")
    print("because momentum persists over days/weeks (Jegadeesh & Titman, 1993).")
    print()
    
    horizons = [1, 3, 5, 10, 20]
    results = []
    
    for horizon in horizons:
        print(f"Training {horizon}-day horizon...")
        result = train_horizon(horizon)
        results.append(result)
        print(f"  → Accuracy: {result['accuracy']:.1%}, AUC: {result['auc']:.3f}")
        print()
    
    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Horizon':>10} | {'Samples':>10} | {'UP %':>8} | {'Accuracy':>10} | {'AUC':>8}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['horizon_days']:>7}d   | {r['samples']:>10,} | {r['up_pct']:>8} | {r['accuracy']:>10.1%} | {r['auc']:>8.3f}")
    
    # Find best
    best = max(results, key=lambda x: x['accuracy'])
    print()
    print(f"Best: {best['horizon_days']}-day horizon with {best['accuracy']:.1%} accuracy (AUC: {best['auc']:.3f})")
    
    # Interpretation
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    # Check if longer horizons are better
    short_acc = results[0]["accuracy"]  # 1-day
    long_acc = results[-1]["accuracy"]  # 20-day
    
    if long_acc > short_acc + 0.02:
        print("✅ Longer horizons show stronger signal — momentum persistence confirmed!")
        print("   Consider focusing on swing trades (5-20 day holds) over day trades.")
    elif abs(long_acc - short_acc) < 0.02:
        print("⚠️ Horizons show similar accuracy — no clear momentum advantage.")
        print("   Technical indicators alone may not capture enough signal.")
    else:
        print("❌ Shorter horizons show stronger signal — unexpected result.")
        print("   May indicate mean reversion dominates in this sample.")
    
    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(RESULTS_DIR, f"horizon_sweep_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to horizon_results/horizon_sweep_{ts}.json")
    
    return results


if __name__ == "__main__":
    run_horizon_sweep()
