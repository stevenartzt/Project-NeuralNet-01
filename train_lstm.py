#!/usr/bin/env python3
"""
LSTM Training — Recurrent model that sees sequences of candles.
Discovers temporal patterns (like divergence) on its own.
"""

import os
import json
import time
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DirectionLSTM(nn.Module):
    """LSTM for directional prediction — sees sequences of candles."""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)


def create_sequences(X, y, seq_length=10):
    """Create sliding window sequences from flat data.
    Each sample becomes a sequence of seq_length consecutive candles.
    """
    sequences = []
    labels = []
    for i in range(seq_length, len(X)):
        sequences.append(X[i - seq_length:i])
        labels.append(y[i])
    return np.array(sequences), np.array(labels)


def load_data(seq_length=10):
    """Load data and create sequences."""
    path = os.path.join(DATA_DIR, "training_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("No training data. Run fetch_data.py first.")

    df = pd.read_csv(path)
    available = [c for c in FEATURE_COLS if c in df.columns]

    # Need to create sequences per ticker (don't mix tickers)
    all_X_seq = []
    all_y_seq = []

    tickers = df["ticker"].unique() if "ticker" in df.columns else ["all"]

    for ticker in tickers:
        if ticker == "all":
            ticker_df = df
        else:
            ticker_df = df[df["ticker"] == ticker].sort_values("Date" if "Date" in df.columns else df.columns[0])

        X = ticker_df[available].values
        y = ticker_df["label"].values

        # Remove NaN rows
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[mask], y[mask]

        if len(X) < seq_length + 10:
            continue

        # Detect binary vs 3-class
        unique_labels = set(np.unique(y).astype(int))
        if unique_labels != {0, 1}:
            y = y + 1  # -1,0,1 → 0,1,2

        # Create sequences for this ticker
        X_seq, y_seq = create_sequences(X, y, seq_length)
        all_X_seq.append(X_seq)
        all_y_seq.append(y_seq)

    X_all = np.concatenate(all_X_seq, axis=0)
    y_all = np.concatenate(all_y_seq, axis=0)

    # Detect num_classes
    unique = set(np.unique(y_all).astype(int))
    num_classes = 2 if unique == {0, 1} else 3

    # Time-ordered split
    split = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    # Scale features (reshape to 2D, scale, reshape back)
    n_train, seq_len, n_feat = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_feat)
    X_test_flat = X_test.reshape(-1, n_feat)
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_test_flat = scaler.transform(X_test_flat)
    X_train = X_train_flat.reshape(n_train, seq_len, n_feat)
    X_test = X_test_flat.reshape(n_test, seq_len, n_feat)

    return X_train, X_test, y_train, y_test, available, scaler, num_classes


def train_lstm(
    seq_length=10,
    hidden_size=128,
    num_layers=2,
    epochs=60,
    batch_size=512,
    lr=0.001,
    dropout=0.2,
    patience=15,
    run_id=None,
):
    if run_id is None:
        run_id = "lstm_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    history_path = os.path.join(MODELS_DIR, f"history_{run_id}.json")
    model_path = os.path.join(MODELS_DIR, f"model_{run_id}.pt")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{run_id}.pkl")

    print(f"🧠 LSTM Training: {run_id}")
    print(f"   Sequence length: {seq_length} candles")
    print(f"   Device: {DEVICE}")
    print("=" * 50)

    # Load data
    X_train, X_test, y_train, y_test, features, scaler, num_classes = load_data(seq_length)
    print(f"Data: {len(X_train)} train, {len(X_test)} test")
    print(f"Sequence shape: {X_train.shape} (samples, candles, features)")
    print(f"Classes: {num_classes} ({'Binary' if num_classes == 2 else '3-class'})")

    # Tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train.astype(int)).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.LongTensor(y_test.astype(int)).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True
    )

    # Class weights
    class_counts = np.bincount(y_train.astype(int), minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * num_classes
    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    # Model
    n_features = X_train.shape[2]
    model = DirectionLSTM(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: LSTM({n_features}→{hidden_size}×{num_layers}) → FC(64) → {num_classes}")
    print(f"Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training history
    history = {
        "run_id": run_id,
        "model_type": "LSTM",
        "config": {
            "seq_length": seq_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "dropout": dropout,
            "features": features,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "num_classes": num_classes,
        },
        "epochs_data": [],
        "status": "training",
        "started_at": datetime.now().isoformat(),
        "network_architecture": {
            "layers": [
                {"type": "linear", "in": n_features * seq_length, "out": hidden_size},
                {"type": "relu"},
                {"type": "linear", "in": hidden_size, "out": 64},
                {"type": "relu"},
                {"type": "linear", "in": 64, "out": num_classes},
            ],
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

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_out = model(X_test_t)
            test_loss = criterion(test_out, y_test_t).item()
            test_preds = test_out.argmax(1)
            test_acc = (test_preds == y_test_t).float().mean().item()

            test_preds_np = test_preds.cpu().numpy()
            y_test_np = y_test_t.cpu().numpy()

            per_class = {}
            labels = ["DOWN", "UP"] if num_classes == 2 else ["SELL", "NEUTRAL", "BUY"]
            for cls_idx, cls_name in enumerate(labels):
                mask = y_test_np == cls_idx
                if mask.sum() > 0:
                    per_class[cls_name] = float((test_preds_np[mask] == cls_idx).mean())

            cm = confusion_matrix(y_test_np, test_preds_np, labels=list(range(num_classes)))

        train_acc = train_correct / train_total
        avg_loss = running_loss / len(train_loader)
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "test_loss": round(test_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "per_class_accuracy": per_class,
            "learning_rate": current_lr,
            "confusion_matrix": cm.tolist(),
            "activation_stats": [],
            "timestamp": datetime.now().isoformat(),
        }
        history["epochs_data"].append(epoch_data)

        # Save history
        try:
            tmp = history_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(history, f, indent=2, default=str)
            os.replace(tmp, history_path)
        except (PermissionError, OSError):
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2, default=str)

        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}/{test_loss:.4f} | "
              f"Acc: {train_acc:.3f}/{test_acc:.3f} | LR: {current_lr:.6f}")

        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "input_size": n_features,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "seq_length": seq_length,
                    "num_classes": num_classes,
                },
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

    # Final report
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t).argmax(1).cpu().numpy()
        final_report = classification_report(
            y_test_t.cpu().numpy(), final_preds,
            target_names=labels, output_dict=True
        )

    history["status"] = "complete"
    history["completed_at"] = datetime.now().isoformat()
    history["best_test_accuracy"] = round(best_acc, 4)
    history["final_report"] = final_report

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"\n✅ LSTM Training complete!")
    print(f"   Best test accuracy: {best_acc:.4f}")
    print(f"   Model: {model_path}")

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LSTM Direction Predictor")
    parser.add_argument("--seq", type=int, default=10, help="Sequence length (candles to look back)")
    parser.add_argument("--hidden", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--layers", type=int, default=2, help="LSTM layers")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID for dashboard")
    args = parser.parse_args()

    train_lstm(
        seq_length=args.seq,
        hidden_size=args.hidden,
        num_layers=args.layers,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        run_id=args.run_id,
    )
