#!/usr/bin/env python3
"""
Train model specifically for mean reversion prediction.
Uses mean_reversion_data.csv (only extreme conditions).
Binary classification: will price revert or continue?
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import pickle

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All features including reversion-relevant ones
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
    "rsi_divergence", "rsi_bull_divergence", "rsi_bear_divergence",
    "rsi_hidden_bull", "rsi_hidden_bear",
]


class ReversionNet(nn.Module):
    """MLP for mean reversion prediction."""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.2):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 2))  # Binary: revert or not
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_reversion_data():
    """Load mean reversion dataset."""
    path = os.path.join(DATA_DIR, "mean_reversion_data.csv")
    if not os.path.exists(path):
        print("❌ No mean reversion data. Run: python fetch_mean_reversion.py")
        sys.exit(1)

    df = pd.read_csv(path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    
    X = df[available].values
    y = df["label"].values.astype(int)  # 0 = no reversion, 1 = reverted
    
    # Remove NaN
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    
    # Time-ordered split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, available, scaler, df


def train_reversion(epochs=60, hidden_sizes=[128, 64, 32], dropout=0.2,
                     lr=0.001, batch_size=256, patience=15, run_id=None):
    """Train mean reversion model."""
    if run_id is None:
        run_id = "reversion_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    history_path = os.path.join(MODELS_DIR, f"history_{run_id}.json")
    model_path = os.path.join(MODELS_DIR, f"model_{run_id}.pt")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{run_id}.pkl")
    
    print(f"🔄 Mean Reversion Training: {run_id}")
    print(f"   Device: {DEVICE}")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test, features, scaler, raw_df = load_reversion_data()
    n_features = len(features)
    
    # Load metadata
    meta_path = os.path.join(DATA_DIR, "mean_reversion_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    
    print(f"Data: {len(X_train)} train, {len(X_test)} test, {n_features} features")
    print(f"Labels: revert={int(y_train.sum())}, no_revert={int(len(y_train) - y_train.sum())}")
    print(f"Reversion rate: {y_train.mean()*100:.1f}%")
    print(f"Settings: RSI <{meta.get('rsi_oversold','?')}/{meta.get('rsi_overbought','?')}>, "
          f"target {meta.get('reversion_target',0)*100}%, lookahead {meta.get('lookahead','?')}")
    
    # Tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.LongTensor(y_test).to(DEVICE)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    
    # Class weights
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 2
    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    
    model = ReversionNet(n_features, hidden_sizes, dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_features} → {' → '.join(map(str, hidden_sizes))} → 2")
    print(f"Parameters: {total_params:,}")
    
    history = {
        "run_id": run_id,
        "model_type": "MeanReversion",
        "config": {
            "epochs": epochs, "hidden_sizes": hidden_sizes, "dropout": dropout,
            "lr": lr, "batch_size": batch_size, "features": features,
            "train_size": len(X_train), "test_size": len(X_test),
            "reversion_settings": meta,
        },
        "epochs_data": [],
        "status": "training",
        "started_at": datetime.now().isoformat(),
        "network_architecture": {
            "layers": [{"type": "linear", "in": n_features, "out": hidden_sizes[0]}] +
                      [{"type": "linear", "in": hidden_sizes[i], "out": hidden_sizes[i+1]} for i in range(len(hidden_sizes)-1)] +
                      [{"type": "linear", "in": hidden_sizes[-1], "out": 2}],
            "total_params": total_params,
        },
    }
    
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        train_correct = 0
        train_total = 0
        
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = out.max(1)
            train_correct += (pred == by).sum().item()
            train_total += by.size(0)
        
        model.eval()
        with torch.no_grad():
            test_out = model(X_test_t)
            test_loss = criterion(test_out, y_test_t).item()
            test_preds = test_out.argmax(1)
            test_acc = (test_preds == y_test_t).float().mean().item()
            
            preds_np = test_preds.cpu().numpy()
            y_np = y_test_t.cpu().numpy()
            per_class = {
                "no_revert": float((preds_np[y_np == 0] == 0).mean()) if (y_np == 0).sum() > 0 else 0,
                "revert": float((preds_np[y_np == 1] == 1).mean()) if (y_np == 1).sum() > 0 else 0,
            }
            cm = confusion_matrix(y_np, preds_np, labels=[0, 1])
        
        train_acc = train_correct / train_total
        avg_loss = running_loss / len(train_loader)
        scheduler.step(test_loss)
        
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "test_loss": round(test_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "per_class_accuracy": per_class,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "confusion_matrix": cm.tolist(),
            "activation_stats": [],
            "timestamp": datetime.now().isoformat(),
        }
        history["epochs_data"].append(epoch_data)
        
        try:
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2, default=str)
        except:
            pass
        
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}/{test_loss:.4f} | "
              f"Acc: {train_acc:.3f}/{test_acc:.3f} | Revert: {per_class['revert']:.3f} | NoRevert: {per_class['no_revert']:.3f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {"input_size": n_features, "hidden_sizes": hidden_sizes, "dropout": dropout},
                "features": features,
                "test_accuracy": test_acc,
            }, model_path)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚡ Early stopping at epoch {epoch+1}")
                break
    
    # Feature importance
    print("\n📊 Computing feature importance...")
    model.eval()
    X_test_t_grad = X_test_t.detach().requires_grad_(True)
    outputs = model(X_test_t_grad)
    outputs.sum().backward()
    importance = X_test_t_grad.grad.abs().mean(0).cpu().numpy()
    importance_dict = {name: round(float(imp), 6) for name, imp in zip(features, importance)}
    importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    history["feature_importance"] = importance_sorted
    
    print("\n🏆 Mean Reversion Feature Importance:")
    for i, (name, imp) in enumerate(list(importance_sorted.items())[:15]):
        bar = "█" * int(imp * 100)
        print(f"   {i+1}. {name:20s} {imp:.4f} {bar}")
    
    # Final report
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t).argmax(1).cpu().numpy()
        final_report = classification_report(
            y_test_t.cpu().numpy(), final_preds,
            target_names=["NO_REVERT", "REVERT"],
            output_dict=True
        )
    
    history["status"] = "complete"
    history["completed_at"] = datetime.now().isoformat()
    history["best_test_accuracy"] = round(best_acc, 4)
    history["final_report"] = final_report
    
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"\n✅ Mean Reversion Training complete!")
    print(f"   Best test accuracy: {best_acc:.4f}")
    print(f"   Model: {model_path}")
    
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mean Reversion Model Training")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()
    
    train_reversion(epochs=args.epochs, lr=args.lr, batch_size=args.batch,
                     dropout=args.dropout, run_id=args.run_id)
