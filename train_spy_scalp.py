#!/usr/bin/env python3
"""
Train LSTM for SPY 0DTE Scalping
Predicts: "Will SPY move 0.1%+ UP in the next 15 minutes?"
Uses intraday-specific features (VWAP, opening range, time of day).
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

SCALP_FEATURES = [
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_position", "bb_width", "atr_pct",
    "stoch_k", "stoch_d",
    "vwap_deviation", "day_range_position",
    "above_or", "below_or", "or_position",
    "time_pct", "hour", "minute",
    "return_1", "return_3", "return_6", "return_12",
    "vol_5", "vol_12", "volume_ratio",
    "consec_up", "consec_down", "body_ratio", "close_in_range",
]


class ScalpLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


def create_sequences(X, y, seq_len=10):
    """Create sliding window sequences — only within same day."""
    seqs, labels = [], []
    for i in range(seq_len, len(X)):
        seqs.append(X[i - seq_len:i])
        labels.append(y[i])
    return np.array(seqs), np.array(labels)


def load_scalp_data(seq_len=10):
    path = os.path.join(DATA_DIR, "spy_scalp_data.csv")
    if not os.path.exists(path):
        print("❌ No scalp data. Run: python fetch_spy_scalp.py")
        sys.exit(1)

    df = pd.read_csv(path)
    available = [c for c in SCALP_FEATURES if c in df.columns]
    
    X = df[available].values.astype(np.float32)
    y = df["label"].values.astype(int)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    X_seq, y_seq = create_sequences(X, y, seq_len)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Scale
    n_tr, sl, nf = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, nf)).reshape(n_tr, sl, nf)
    X_test = scaler.transform(X_test.reshape(-1, nf)).reshape(len(X_test), sl, nf)

    return X_train, X_test, y_train, y_test, available, scaler


def train_spy_scalp(seq_len=10, hidden_size=64, num_layers=2, epochs=60,
                     batch_size=256, lr=0.001, dropout=0.3, patience=15, run_id=None):
    if run_id is None:
        run_id = "spy_scalp_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    history_path = os.path.join(MODELS_DIR, f"history_{run_id}.json")
    model_path = os.path.join(MODELS_DIR, f"model_{run_id}.pt")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{run_id}.pkl")

    print(f"📈 SPY 0DTE Scalp LSTM: {run_id}")
    print(f"   Device: {DEVICE}")
    print(f"   Sequence: {seq_len} bars ({seq_len * 5} min lookback)")
    print("=" * 50)

    X_train, X_test, y_train, y_test, features, scaler = load_scalp_data(seq_len)
    n_features = X_train.shape[2]

    print(f"Data: {len(X_train)} train, {len(X_test)} test, {n_features} features")
    print(f"Labels: UP={int(y_train.sum())} ({y_train.mean()*100:.1f}%), DOWN={int(len(y_train)-y_train.sum())}")

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.LongTensor(y_train).to(DEVICE)
    X_te = torch.FloatTensor(X_test).to(DEVICE)
    y_te = torch.LongTensor(y_test).to(DEVICE)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    # Class weights
    counts = np.bincount(y_train, minlength=2)
    weights = torch.FloatTensor((1.0 / (counts + 1)) / (1.0 / (counts + 1)).sum() * 2).to(DEVICE)

    model = ScalpLSTM(n_features, hidden_size, num_layers, dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: LSTM({n_features}→{hidden_size}×{num_layers}) → 32 → 2")
    print(f"Parameters: {params:,}")

    history = {
        "run_id": run_id, "model_type": "SPY_Scalp_LSTM",
        "config": {
            "seq_len": seq_len, "hidden_size": hidden_size, "num_layers": num_layers,
            "epochs": epochs, "lr": lr, "dropout": dropout, "batch_size": batch_size,
            "features": features, "train_size": len(X_train), "test_size": len(X_test),
        },
        "epochs_data": [], "status": "training",
        "started_at": datetime.now().isoformat(),
        "network_architecture": {
            "layers": [
                {"type": "linear", "in": n_features * seq_len, "out": hidden_size},
                {"type": "relu"},
                {"type": "linear", "in": hidden_size, "out": 32},
                {"type": "relu"},
                {"type": "linear", "in": 32, "out": 2},
            ],
            "total_params": params,
        },
    }

    best_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == by).sum().item()
            total += by.size(0)

        model.eval()
        with torch.no_grad():
            te_out = model(X_te)
            te_loss = criterion(te_out, y_te).item()
            te_preds = te_out.argmax(1)
            te_acc = (te_preds == y_te).float().mean().item()

            preds_np = te_preds.cpu().numpy()
            y_np = y_te.cpu().numpy()
            per_class = {
                "DOWN": float((preds_np[y_np == 0] == 0).mean()) if (y_np == 0).sum() > 0 else 0,
                "UP": float((preds_np[y_np == 1] == 1).mean()) if (y_np == 1).sum() > 0 else 0,
            }
            cm = confusion_matrix(y_np, preds_np, labels=[0, 1])

        train_acc = correct / total
        avg_loss = total_loss / len(loader)
        scheduler.step(te_loss)

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6), "test_loss": round(te_loss, 6),
            "train_accuracy": round(train_acc, 4), "test_accuracy": round(te_acc, 4),
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
        except: pass

        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}/{te_loss:.4f} | "
              f"Acc: {train_acc:.3f}/{te_acc:.3f} | UP:{per_class['UP']:.3f} DOWN:{per_class['DOWN']:.3f}")

        if te_acc > best_acc:
            best_acc = te_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {"input_size": n_features, "hidden_size": hidden_size,
                           "num_layers": num_layers, "dropout": dropout, "seq_len": seq_len},
                "features": features, "test_accuracy": te_acc,
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
    model.train()
    sample = min(3000, len(X_te))
    X_s = X_te[:sample].detach().requires_grad_(True)
    out = model(X_s)
    out.sum().backward()
    grads = X_s.grad.abs().mean(dim=0).mean(dim=0).cpu().numpy()
    importance = {name: round(float(g), 6) for name, g in zip(features, grads)}
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    history["feature_importance"] = importance

    print("\n🏆 SPY Scalp Feature Importance:")
    for i, (name, imp) in enumerate(list(importance.items())[:15]):
        bar = "█" * int(imp * 200)
        print(f"   {i+1}. {name:20s} {imp:.4f} {bar}")

    # Final
    model.eval()
    with torch.no_grad():
        fp = model(X_te).argmax(1).cpu().numpy()
        report = classification_report(y_te.cpu().numpy(), fp, target_names=["DOWN", "UP"], output_dict=True)

    history["status"] = "complete"
    history["completed_at"] = datetime.now().isoformat()
    history["best_test_accuracy"] = round(best_acc, 4)
    history["final_report"] = report
    history["feature_importance"] = importance

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\n✅ SPY Scalp Training complete!")
    print(f"   Best accuracy: {best_acc:.4f}")
    print(f"   Model: {model_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SPY 0DTE Scalp LSTM")
    parser.add_argument("--seq", type=int, default=10, help="Lookback bars (default: 10 = 50 min)")
    parser.add_argument("--hidden", type=int, default=64, help="LSTM hidden size")
    parser.add_argument("--layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--run-id", type=str, default=None)
    args = parser.parse_args()

    train_spy_scalp(
        seq_len=args.seq, hidden_size=args.hidden, num_layers=args.layers,
        epochs=args.epochs, lr=args.lr, dropout=args.dropout,
        batch_size=args.batch, run_id=args.run_id,
    )
