#!/usr/bin/env python3
"""
Auto-Optimizer — Bayesian hyperparameter search.
Runs hundreds of training iterations, automatically finding the best architecture.
Uses Optuna (Tree-structured Parzen Estimator) to intelligently explore the search space.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
import optuna
from optuna.trial import Trial

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OPTIMIZE_DIR = os.path.join(os.path.dirname(__file__), "optimize")
os.makedirs(OPTIMIZE_DIR, exist_ok=True)

# Feature columns
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


def load_data():
    """Load and prepare data."""
    path = os.path.join(DATA_DIR, "training_data.csv")
    df = pd.read_csv(path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df["label"].values + 1  # -1,0,1 → 0,1,2

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    # Time-ordered split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, available, scaler


class FlexNet(nn.Module):
    """Flexible architecture neural net."""
    def __init__(self, input_size, hidden_sizes, dropout, activation='relu'):
        super().__init__()
        layers = []
        prev = input_size
        act_fn = nn.ReLU if activation == 'relu' else nn.LeakyReLU if activation == 'leaky_relu' else nn.GELU

        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                act_fn(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_trial(trial: Trial, X_train, X_test, y_train, y_test, n_features, study_id):
    """Single training trial with Optuna-suggested hyperparameters."""

    # ── Hyperparameter search space ──
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_sizes = []
    for i in range(n_layers):
        h = trial.suggest_int(f"hidden_{i}", 16, 128, step=16)
        hidden_sizes.append(h)

    dropout = trial.suggest_float("dropout", 0.05, 0.4, step=0.05)
    lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "gelu"])
    epochs = 20  # Aggressive — early stopping catches the rest

    # ── Build model ──
    model = FlexNet(n_features, hidden_sizes, dropout, activation).to(DEVICE)

    # Class weights
    class_counts = np.bincount(y_train.astype(int), minlength=3)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 3
    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train.astype(int)).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.LongTensor(y_test.astype(int)).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True
    )

    best_acc = 0
    patience_counter = 0
    patience = 6

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(X_test_t).argmax(1).cpu().numpy()
            acc = accuracy_score(y_test_t.cpu().numpy(), preds)

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        # Optuna pruning — stop bad trials early
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_acc


def run_tree_baselines(X_train, X_test, y_train, y_test, features):
    """Run fast tree-based models as accuracy baselines."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    results = {}

    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                            eval_metric='mlogloss',
                            n_jobs=1, verbosity=0)
        xgb.fit(X_train, y_train)
        results["XGBoost"] = round(accuracy_score(y_test, xgb.predict(X_test)), 4)
    except Exception as e:
        print(f"   XGBoost failed: {e}")

    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                              n_jobs=1, verbose=-1)
        lgbm.fit(X_train, y_train)
        results["LightGBM"] = round(accuracy_score(y_test, lgbm.predict(X_test)), 4)
    except Exception as e:
        print(f"   LightGBM failed: {e}")

    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=1, random_state=42)
        rf.fit(X_train, y_train)
        results["RandomForest"] = round(accuracy_score(y_test, rf.predict(X_test)), 4)

        # Feature importance from RF
        importance = dict(zip(features, rf.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        results["_rf_importance"] = {k: round(v, 4) for k, v in list(importance.items())[:10]}
    except Exception as e:
        print(f"   RandomForest failed: {e}")

    try:
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        results["GradientBoosting"] = round(accuracy_score(y_test, gb.predict(X_test)), 4)
    except Exception as e:
        print(f"   GradientBoosting failed: {e}")

    return results


def run_optimization(n_trials=200, study_id=None):
    """Run full hyperparameter optimization."""
    if study_id is None:
        study_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    status_path = os.path.join(OPTIMIZE_DIR, f"status_{study_id}.json")
    best_path = os.path.join(OPTIMIZE_DIR, f"best_{study_id}.json")

    print(f"🧬 Auto-Optimizer — {n_trials} trials")
    print(f"   Study ID: {study_id}")
    print(f"   Device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print("=" * 50)

    # Load data once
    X_train, X_test, y_train, y_test, features, scaler = load_data()
    n_features = len(features)
    print(f"Data: {len(X_train)} train, {len(X_test)} test, {n_features} features")

    # Run tree-based baselines first (seconds, not minutes)
    print("\n🌲 Running tree-based baselines...")
    tree_results = run_tree_baselines(X_train, X_test, y_train, y_test, features)
    for name, acc in tree_results.items():
        if not name.startswith("_"):
            print(f"   {name}: {acc:.4f}")

    # Create Optuna study with aggressive pruning
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name=f"direction_net_{study_id}"
    )

    start_time = time.time()

    def objective(trial):
        trial_start = time.time()
        acc = train_trial(trial, X_train, X_test, y_train, y_test, n_features, study_id)
        trial_time = time.time() - trial_start

        # Update status file for dashboard
        elapsed = time.time() - start_time
        completed = trial.number + 1
        trials_data = []
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        for t in complete_trials:
            trials_data.append({
                "number": t.number,
                "accuracy": round(t.value, 4),
                "params": t.params,
                "duration": round(t.duration.total_seconds(), 1) if t.duration else 0,
            })

        best_val = study.best_value if complete_trials else acc
        best_params = study.best_params if complete_trials else trial.params

        status = {
            "study_id": study_id,
            "status": "running",
            "total_trials": n_trials,
            "completed_trials": completed,
            "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "best_accuracy": round(best_val, 4),
            "best_params": best_params,
            "elapsed_seconds": round(elapsed, 1),
            "avg_trial_seconds": round(elapsed / max(completed, 1), 1),
            "est_remaining_seconds": round((elapsed / max(completed, 1)) * (n_trials - completed), 1),
            "current_trial": {
                "number": trial.number,
                "accuracy": round(acc, 4),
                "params": trial.params,
                "duration": round(trial_time, 1),
            },
            "all_trials": trials_data[-50:],  # Last 50 for chart
            "accuracy_over_time": [round(t.value, 4) for t in complete_trials],
            "updated_at": datetime.now().isoformat(),
            "tree_baselines": {k: v for k, v in tree_results.items() if not k.startswith("_")},
            "tree_feature_importance": tree_results.get("_rf_importance", {}),
        }

        with open(status_path + ".tmp", "w") as f:
            json.dump(status, f, indent=2, default=str)
        os.replace(status_path + ".tmp", status_path)

        print(f"  Trial {completed}/{n_trials} | Acc: {acc:.4f} | Best: {best_val:.4f} | Time: {trial_time:.1f}s | Params: {dict(list(trial.params.items())[:3])}...")

        return acc

    # Run optimization (2 parallel jobs for our 2 CPUs)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    # Save best model
    print(f"\n{'='*50}")
    print(f"✅ Optimization complete!")
    print(f"   Best accuracy: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")
    print(f"   Total time: {time.time() - start_time:.0f}s")

    # Train final model with best params
    print("\n🏆 Training final model with best parameters...")
    best = study.best_params
    n_layers = best["n_layers"]
    hidden_sizes = [best[f"hidden_{i}"] for i in range(n_layers)]

    model = FlexNet(n_features, hidden_sizes, best["dropout"], best["activation"]).to(DEVICE)
    weights_tensor = torch.FloatTensor([1.0, 1.0, 1.0])
    class_counts = np.bincount(y_train.astype(int), minlength=3)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 3
    weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=best["lr"], weight_decay=best["weight_decay"])

    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.LongTensor(y_train.astype(int)).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.LongTensor(y_test.astype(int)).to(DEVICE)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=best["batch_size"], shuffle=True)

    best_acc = 0
    for epoch in range(60):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = (model(X_test_t).argmax(1) == y_test_t).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            import pickle
            torch.save({
                "model_state": model.state_dict(),
                "config": {"input_size": n_features, "hidden_sizes": hidden_sizes, "dropout": best["dropout"]},
                "features": features,
                "test_accuracy": acc,
                "epoch": epoch + 1,
                "optimization": {"study_id": study_id, "n_trials": n_trials, "best_params": best}
            }, os.path.join(MODELS_DIR, f"model_optimized_{study_id}.pt"))
            with open(os.path.join(MODELS_DIR, f"scaler_optimized_{study_id}.pkl"), "wb") as f:
                pickle.dump(scaler, f)

    print(f"   Final best accuracy: {best_acc:.4f}")
    print(f"   Model saved: model_optimized_{study_id}.pt")

    # Update status to complete
    with open(status_path) as f:
        status = json.load(f)
    status["status"] = "complete"
    status["final_accuracy"] = round(best_acc, 4)
    status["best_params"] = best
    status["completed_at"] = datetime.now().isoformat()
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2, default=str)

    # Save best config for easy reference
    with open(best_path, "w") as f:
        json.dump({
            "accuracy": round(best_acc, 4),
            "params": best,
            "hidden_sizes": hidden_sizes,
            "features": features,
            "study_id": study_id,
            "n_trials": n_trials,
        }, f, indent=2)

    # Run a full training with best params so dashboard gets loss curves, activations, etc.
    print("\n📊 Running full training with best params for dashboard visualization...")
    from train import train_model
    train_model(
        epochs=60,
        batch_size=best.get("batch_size", 128),
        lr=best.get("lr", 0.001),
        hidden_sizes=hidden_sizes,
        dropout=best.get("dropout", 0.1),
        patience=15,
        run_id=f"optimized_{study_id}"
    )
    print(f"   Dashboard run saved as: optimized_{study_id}")

    return study


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    study_id = os.environ.get("STUDY_ID", None)
    run_optimization(n_trials=n, study_id=study_id)
