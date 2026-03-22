#!/usr/bin/env python3
"""
Binary Direction Classifier — UP vs DOWN
Simpler problem: predicts whether stock goes up or down (ignores neutral).
This tests whether there's any predictive signal in the features.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from datetime import datetime
import pickle

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

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


class BinaryNet(nn.Module):
    """Binary classifier: UP (1) or DOWN (0)"""
    def __init__(self, input_size, hidden_sizes=[64, 32], dropout=0.3):
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
        layers.append(nn.Linear(prev_size, 1))  # Binary output
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def load_binary_data(threshold=0.005):
    """
    Load data with binary labels.
    UP: future_return > threshold
    DOWN: future_return < -threshold
    Neutral samples are dropped.
    """
    path = os.path.join(DATA_DIR, "training_data.csv")
    df = pd.read_csv(path)
    
    # Create binary labels (drop neutral)
    df = df.dropna(subset=['future_return'])
    up = df['future_return'] > threshold
    down = df['future_return'] < -threshold
    
    # Keep only clear signals
    df = df[up | down].copy()
    df['binary_label'] = (df['future_return'] > threshold).astype(int)
    
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df['binary_label'].values
    
    # Handle NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, available


def train_binary(epochs=100, batch_size=128, lr=0.0005, threshold=0.005, verbose=True):
    """Train binary classifier."""
    print(f"Loading data with threshold={threshold}...")
    X, y, features = load_binary_data(threshold)
    
    print(f"Total samples: {len(y):,}")
    print(f"UP: {y.sum():,} ({y.mean():.1%})")
    print(f"DOWN: {(1-y).sum():,} ({1-y.mean():.1%})")
    print(f"Features: {len(features)}")
    print()
    
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
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    # Model
    model = BinaryNet(input_size=len(features), hidden_sizes=[64, 32], dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}
    
    print("Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t).squeeze()
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)
        
        train_preds = np.array(train_preds)
        train_acc = accuracy_score(train_labels, (train_preds > 0.5).astype(int))
        val_acc = accuracy_score(y_test, val_preds)
        val_auc = roc_auc_score(y_test, val_probs)
        
        history["train_loss"].append(total_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        
        scheduler.step(1 - val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d} | Train: {train_acc:.1%} | Val: {val_acc:.1%} | AUC: {val_auc:.3f}")
    
    # Final evaluation with best model
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_t).squeeze()
        val_probs = torch.sigmoid(val_outputs).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)
    
    print()
    print("=" * 50)
    print(f"Best Validation Accuracy: {best_val_acc:.1%} (epoch {best_epoch})")
    print(f"Final AUC: {roc_auc_score(y_test, val_probs):.3f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, val_preds, target_names=["DOWN", "UP"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, val_preds))
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"binary_{ts}.pt"))
    with open(os.path.join(MODELS_DIR, f"binary_scaler_{ts}.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    
    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["threshold"] = threshold
    history["features"] = features
    with open(os.path.join(MODELS_DIR, f"binary_history_{ts}.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    return best_val_acc, history


def sweep_thresholds():
    """Try different labeling thresholds to find optimal signal."""
    print("=" * 60)
    print("THRESHOLD SWEEP — Finding optimal signal definition")
    print("=" * 60)
    print()
    
    results = []
    for threshold in [0.001, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]:
        print(f"\n--- Threshold: {threshold:.1%} ---")
        acc, history = train_binary(
            epochs=50, threshold=threshold, verbose=False
        )
        results.append({
            "threshold": threshold,
            "accuracy": acc,
            "auc": max(history["val_auc"])
        })
        print(f"  Result: Acc={acc:.1%}, AUC={max(history['val_auc']):.3f}")
    
    print("\n" + "=" * 60)
    print("SWEEP RESULTS")
    print("=" * 60)
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"  Threshold {r['threshold']:.1%}: Acc={r['accuracy']:.1%}, AUC={r['auc']:.3f}")
    
    best = max(results, key=lambda x: x["accuracy"])
    print(f"\nBest: {best['threshold']:.1%} with {best['accuracy']:.1%} accuracy")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sweep":
        sweep_thresholds()
    else:
        # Single training run
        train_binary(epochs=100, threshold=0.005)
