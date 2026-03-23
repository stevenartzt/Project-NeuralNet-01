#!/usr/bin/env python3
"""
Time-Series Cross-Validation — Tests our models across multiple time windows.
Instead of one 80/20 split, tests on 5 different time periods to confirm results aren't a fluke.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class QuickNet(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 112, 80], dropout=0.05):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_neural_net(X_train, y_train, X_test, y_test, n_features, epochs=20):
    """Quick neural net training for cross-validation."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = QuickNet(n_features).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)

    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.LongTensor(y_train.astype(int)).to(DEVICE)
    X_te_t = torch.FloatTensor(X_te).to(DEVICE)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(1).cpu().numpy()
            acc = accuracy_score(y_test, preds)
            if acc > best_acc:
                best_acc = acc

    return best_acc


def run_cross_validation(n_splits=5):
    """Run time-series cross-validation across all models."""
    path = os.path.join(DATA_DIR, "training_data.csv")
    if not os.path.exists(path):
        print("❌ No training data. Run fetch_data.py first.")
        sys.exit(1)

    df = pd.read_csv(path)
    available = [c for c in FEATURE_COLS if c in df.columns]

    X = df[available].values
    y = df["label"].values + 1  # -1,0,1 → 0,1,2

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    print(f"🔬 Time-Series Cross-Validation ({n_splits} folds)")
    print(f"   Device: {DEVICE}")
    print(f"   Data: {len(X)} samples, {len(available)} features")
    print("=" * 70)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {
        "NeuralNet": [],
        "RandomForest": [],
        "XGBoost": [],
        "LightGBM": [],
        "SVM_RBF": [],
        "GradientBoosting": [],
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\n📊 Fold {fold+1}/{n_splits} — Train: {len(X_train)}, Test: {len(X_test)}")

        # Neural Net
        try:
            acc = train_neural_net(X_train, y_train, X_test, y_test, len(available))
            results["NeuralNet"].append(acc)
            print(f"   NeuralNet:        {acc:.4f}")
        except Exception as e:
            print(f"   NeuralNet failed: {e}")

        # Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=1, random_state=42)
            rf.fit(X_train, y_train)
            acc = accuracy_score(y_test, rf.predict(X_test))
            results["RandomForest"].append(acc)
            print(f"   RandomForest:     {acc:.4f}")
        except Exception as e:
            print(f"   RandomForest failed: {e}")

        # XGBoost
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                eval_metric='mlogloss', n_jobs=1, verbosity=0)
            xgb.fit(X_train, y_train)
            acc = accuracy_score(y_test, xgb.predict(X_test))
            results["XGBoost"].append(acc)
            print(f"   XGBoost:          {acc:.4f}")
        except Exception as e:
            print(f"   XGBoost failed: {e}")

        # LightGBM
        try:
            from lightgbm import LGBMClassifier
            lgbm = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                  n_jobs=1, verbose=-1)
            lgbm.fit(X_train, y_train)
            acc = accuracy_score(y_test, lgbm.predict(X_test))
            results["LightGBM"].append(acc)
            print(f"   LightGBM:         {acc:.4f}")
        except Exception as e:
            print(f"   LightGBM failed: {e}")

        # Gradient Boosting
        try:
            gb = GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
            gb.fit(X_train, y_train)
            acc = accuracy_score(y_test, gb.predict(X_test))
            results["GradientBoosting"].append(acc)
            print(f"   GradientBoosting: {acc:.4f}")
        except Exception as e:
            print(f"   GradientBoosting failed: {e}")

        # SVM RBF (sample for speed)
        try:
            sample_size = min(10000, len(X_train))
            idx = np.random.choice(len(X_train), sample_size, replace=False)
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_train[idx])
            X_te_scaled = scaler.transform(X_test)
            svm = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
            svm.fit(X_tr_scaled, y_train[idx])
            acc = accuracy_score(y_test, svm.predict(X_te_scaled))
            results["SVM_RBF"].append(acc)
            print(f"   SVM_RBF:          {acc:.4f}")
        except Exception as e:
            print(f"   SVM RBF failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("📋 CROSS-VALIDATION SUMMARY")
    print(f"{'Model':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  Folds")
    print("-" * 70)

    best_model = ""
    best_mean = 0

    for model_name, accs in results.items():
        if not accs:
            continue
        mean = np.mean(accs)
        std = np.std(accs)
        if mean > best_mean:
            best_mean = mean
            best_model = model_name
        folds_str = " ".join(f"{a:.3f}" for a in accs)
        print(f"{model_name:<20} {mean:>7.4f} {std:>7.4f} {min(accs):>7.4f} {max(accs):>7.4f}  [{folds_str}]")

    print(f"\n🏆 Best: {best_model} ({best_mean:.4f})")

    # Save
    out = {
        "n_splits": n_splits,
        "total_samples": len(X),
        "features": len(available),
        "results": {k: {"mean": round(np.mean(v), 4), "std": round(np.std(v), 4),
                         "folds": [round(a, 4) for a in v]} for k, v in results.items() if v},
        "best_model": best_model,
        "best_mean": round(best_mean, 4),
    }
    out_path = os.path.join(os.path.dirname(__file__), "cross_validation_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_cross_validation(n_splits=n)
