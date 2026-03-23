#!/usr/bin/env python3
"""
Neural Net Training Engine — Direction Predictor (Network 1)
Completely independent. Learns from raw indicators → BUY/SELL/NEUTRAL.
Saves training history for live dashboard visualization.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import pickle

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Feature columns — pure technical indicators
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
]

# Columns excluded from features (raw price data, identifiers, targets)
EXCLUDE_COLS = [
    "Date", "ticker", "label", "future_return",
    "Open", "High", "Low", "Close", "Volume",
    "sma_5", "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
    "bb_upper", "bb_lower", "volume_sma_20", "obv", "obv_sma", "vwap"
]


class DirectionNet(nn.Module):
    """
    MLP for directional prediction.
    Input: technical indicators → Output: BUY/SELL/NEUTRAL probabilities
    """
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3, num_classes=3):
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

        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_activations(self, x):
        """Return activations at each layer for visualization."""
        activations = []
        current = x
        for layer in self.net:
            current = layer(current)
            if isinstance(layer, nn.ReLU):
                activations.append(current.detach().cpu().numpy())
        return activations


def load_data():
    """Load preprocessed training data."""
    path = os.path.join(DATA_DIR, "training_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("No training data. Run fetch_data.py first.")

    df = pd.read_csv(path)

    # Use defined feature columns that exist in data
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    X = df[available_features].values
    y = df["label"].values

    # Remap labels: -1,0,1 → 0,1,2 for cross-entropy
    y = y + 1  # -1→0, 0→1, 1→2

    # Remove any remaining NaN/inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    return X, y, available_features, df


def train_model(
    epochs=100,
    batch_size=256,
    lr=0.001,
    hidden_sizes=[128, 64, 32],
    dropout=0.3,
    patience=15,
    run_id=None
):
    """
    Train the direction prediction model.
    Saves training history to JSON for live dashboard updates.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    history_path = os.path.join(MODELS_DIR, f"history_{run_id}.json")
    model_path = os.path.join(MODELS_DIR, f"model_{run_id}.pt")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{run_id}.pkl")

    print(f"🧠 Training run: {run_id}")
    print("=" * 50)

    # Load data
    X, y, feature_names, raw_df = load_data()
    print(f"Data: {len(X)} samples, {len(feature_names)} features")
    print(f"Classes: sell={np.sum(y==0)}, neutral={np.sum(y==1)}, buy={np.sum(y==2)}")

    # Train/test split (time-aware: last 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Device selection (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train.astype(int)).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test.astype(int)).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train.astype(int), minlength=3)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 3
    weights_tensor = torch.FloatTensor(class_weights)

    # Model
    model = DirectionNet(
        input_size=len(feature_names),
        hidden_sizes=hidden_sizes,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training history
    history = {
        "run_id": run_id,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "features": feature_names,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "class_distribution": {
                "train": {"sell": int(np.sum(y_train==0)), "neutral": int(np.sum(y_train==1)), "buy": int(np.sum(y_train==2))},
                "test": {"sell": int(np.sum(y_test==0)), "neutral": int(np.sum(y_test==1)), "buy": int(np.sum(y_test==2))}
            }
        },
        "epochs_data": [],
        "status": "training",
        "started_at": datetime.now().isoformat(),
        "network_architecture": {
            "layers": [],
            "total_params": sum(p.numel() for p in model.parameters())
        }
    }

    # Record architecture for visualization
    for name, layer in model.net.named_children():
        if isinstance(layer, nn.Linear):
            history["network_architecture"]["layers"].append({
                "type": "linear",
                "in": layer.in_features,
                "out": layer.out_features
            })
        elif isinstance(layer, nn.Dropout):
            history["network_architecture"]["layers"].append({
                "type": "dropout",
                "rate": layer.p
            })
        elif isinstance(layer, nn.ReLU):
            history["network_architecture"]["layers"].append({"type": "relu"})
        elif isinstance(layer, nn.BatchNorm1d):
            history["network_architecture"]["layers"].append({
                "type": "batchnorm",
                "features": layer.num_features
            })

    # Save initial state
    _save_history(history, history_path)

    best_test_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t).item()
            _, test_preds = test_outputs.max(1)
            test_acc = (test_preds == y_test_t).float().mean().item()

            # Per-class accuracy
            test_preds_np = test_preds.numpy()
            y_test_np = y_test_t.numpy()
            per_class = {}
            for cls_idx, cls_name in enumerate(["sell", "neutral", "buy"]):
                mask = y_test_np == cls_idx
                if mask.sum() > 0:
                    per_class[cls_name] = float((test_preds_np[mask] == cls_idx).mean())
                else:
                    per_class[cls_name] = 0.0

            # Get activations for visualization (sample of 100)
            sample_idx = np.random.choice(len(X_test_t), min(100, len(X_test_t)), replace=False)
            sample_activations = model.get_activations(X_test_t[sample_idx])
            activation_stats = []
            for act in sample_activations:
                activation_stats.append({
                    "mean": float(np.mean(act)),
                    "std": float(np.std(act)),
                    "sparsity": float((act == 0).mean()),
                    "max": float(np.max(act)),
                    "per_neuron_mean": np.mean(act, axis=0).tolist()
                })

            # Confusion matrix
            cm = confusion_matrix(y_test_np, test_preds_np, labels=[0,1,2])

        train_acc = train_correct / train_total
        avg_loss = running_loss / len(train_loader)

        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log epoch
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "test_loss": round(test_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "per_class_accuracy": per_class,
            "learning_rate": current_lr,
            "activation_stats": activation_stats,
            "confusion_matrix": cm.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        history["epochs_data"].append(epoch_data)

        # Save history every epoch (dashboard reads this)
        _save_history(history, history_path)

        # Small delay so dashboard can poll each epoch visually
        time.sleep(0.5)

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Loss: {avg_loss:.4f}/{test_loss:.4f} | "
              f"Acc: {train_acc:.3f}/{test_acc:.3f} | "
              f"LR: {current_lr:.6f}")

        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "input_size": len(feature_names),
                    "hidden_sizes": hidden_sizes,
                    "dropout": dropout
                },
                "features": feature_names,
                "test_accuracy": test_acc,
                "epoch": epoch + 1
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚡ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t).argmax(1).numpy()
        final_report = classification_report(
            y_test_t.numpy(), final_preds,
            target_names=["SELL", "NEUTRAL", "BUY"],
            output_dict=True
        )

    history["status"] = "complete"
    history["completed_at"] = datetime.now().isoformat()
    history["best_test_accuracy"] = round(best_test_acc, 4)
    history["final_report"] = final_report
    history["model_path"] = model_path

    # Feature importance via gradient-based attribution
    model.eval()
    X_test_t.requires_grad_(True)
    outputs = model(X_test_t)
    outputs.sum().backward()
    importance = X_test_t.grad.abs().mean(0).numpy()
    importance_dict = {name: round(float(imp), 6) for name, imp in zip(feature_names, importance)}
    importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    history["feature_importance"] = importance_sorted

    _save_history(history, history_path)

    print(f"\n✅ Training complete!")
    print(f"   Best test accuracy: {best_test_acc:.4f}")
    print(f"   Model saved: {model_path}")
    print(f"   Top features: {list(importance_sorted.keys())[:5]}")

    return history


def _save_history(history, path):
    """Save history."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(history, f, indent=2, default=str)
        os.replace(tmp, path)
    except (PermissionError, OSError):
        # Windows file locking — write directly
        with open(path, "w") as f:
            json.dump(history, f, indent=2, default=str)


if __name__ == "__main__":
    run_id = os.environ.get("RUN_ID", None)
    epochs = int(os.environ.get("EPOCHS", 100))
    train_model(epochs=epochs, batch_size=256, lr=0.001, run_id=run_id)
