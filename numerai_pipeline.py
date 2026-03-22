#!/usr/bin/env python3
"""
Numerai Tournament Pipeline
Downloads Numerai data, trains our model on it, submits predictions.
Run weekly (Saturday) for passive income.

Setup:
1. Sign up at https://numer.ai/tournament
2. Get API keys from https://numer.ai/account → API Keys
3. Set environment variables:
   export NUMERAI_PUBLIC_ID="your_public_id"
   export NUMERAI_SECRET_KEY="your_secret_key"
4. python numerai_pipeline.py --setup     # First time: download data + train
5. python numerai_pipeline.py --submit    # Weekly: predict + submit
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
from sklearn.metrics import mean_squared_error
from datetime import datetime
import pickle

try:
    import numerapi
except ImportError:
    print("Install numerapi: pip install numerapi")
    sys.exit(1)

BASE_DIR = os.path.dirname(__file__)
NUMERAI_DIR = os.path.join(BASE_DIR, "numerai_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(NUMERAI_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NumeraiNet(nn.Module):
    """Neural net for Numerai tournament predictions."""
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.1):
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
        layers.append(nn.Linear(prev, 1))  # Regression: predict ranking
        layers.append(nn.Sigmoid())  # Output 0-1
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()


def download_data():
    """Download current Numerai tournament data."""
    napi = numerapi.NumerAPI()
    
    print("📥 Downloading Numerai dataset...")
    current_round = napi.get_current_round()
    print(f"   Current round: {current_round}")
    
    # Download training data
    train_path = os.path.join(NUMERAI_DIR, "train.parquet")
    if not os.path.exists(train_path):
        print("   Downloading training data (this may take a while)...")
        napi.download_dataset("v5.0/train.parquet", dest_path=train_path)
    else:
        print("   Training data already downloaded")
    
    # Download live data for predictions
    live_path = os.path.join(NUMERAI_DIR, "live.parquet")
    print("   Downloading live data...")
    napi.download_dataset("v5.0/live.parquet", dest_path=live_path)
    
    # Download validation data
    val_path = os.path.join(NUMERAI_DIR, "validation.parquet")
    if not os.path.exists(val_path):
        print("   Downloading validation data...")
        napi.download_dataset("v5.0/validation.parquet", dest_path=val_path)
    
    print(f"✅ Data downloaded to {NUMERAI_DIR}")
    return current_round


def load_numerai_data(sample_frac=0.1):
    """Load and prepare Numerai data."""
    train_path = os.path.join(NUMERAI_DIR, "train.parquet")
    
    print("📊 Loading Numerai data...")
    df = pd.read_parquet(train_path)
    print(f"   Full dataset: {len(df)} rows")
    
    # Sample for speed (Numerai data can be millions of rows)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"   Sampled to: {len(df)} rows ({sample_frac*100:.0f}%)")
    
    # Get feature columns (Numerai names them feature_*)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    target_col = "target"
    
    if target_col not in df.columns:
        # Try alternative target names
        target_cols = [c for c in df.columns if c.startswith("target")]
        if target_cols:
            target_col = target_cols[0]
            print(f"   Using target: {target_col}")
        else:
            raise ValueError("No target column found")
    
    print(f"   Features: {len(feature_cols)}")
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    
    # Remove NaN
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    
    # Time-ordered split (use era column if available)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    return X_train, X_val, y_train, y_val, feature_cols


def train_numerai_model(epochs=30, hidden_sizes=[256, 128, 64], lr=0.001, 
                         dropout=0.1, batch_size=4096, sample_frac=0.1):
    """Train model on Numerai data."""
    X_train, X_val, y_train, y_val, features = load_numerai_data(sample_frac)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True
    )
    
    # Model
    model = NumeraiNet(len(features), hidden_sizes, dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    print(f"\n🧠 Training Numerai model on {DEVICE}")
    print(f"   {len(X_train)} train / {len(X_val)} val / {len(features)} features")
    print(f"   Architecture: {len(features)} → {' → '.join(map(str, hidden_sizes))} → 1")
    
    best_corr = -1
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
        
        # Evaluate with correlation (Numerai's scoring metric)
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).cpu().numpy()
            val_true = y_val_t.cpu().numpy()
            corr = np.corrcoef(val_preds, val_true)[0, 1]
            mse = mean_squared_error(val_true, val_preds)
        
        scheduler.step(mse)
        
        if corr > best_corr:
            best_corr = corr
            # Save best model
            torch.save({
                "model_state": model.state_dict(),
                "config": {"input_size": len(features), "hidden_sizes": hidden_sizes, "dropout": dropout},
                "features": features,
                "correlation": corr,
            }, os.path.join(MODELS_DIR, "numerai_model.pt"))
            with open(os.path.join(MODELS_DIR, "numerai_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)
        
        print(f"  Epoch {epoch+1}/{epochs} | Corr: {corr:.4f} | MSE: {mse:.6f} | Best: {best_corr:.4f}")
    
    print(f"\n✅ Training complete! Best correlation: {best_corr:.4f}")
    return best_corr


def generate_predictions():
    """Generate predictions on live Numerai data."""
    live_path = os.path.join(NUMERAI_DIR, "live.parquet")
    if not os.path.exists(live_path):
        print("❌ No live data. Run with --setup first.")
        return None
    
    # Load model
    model_path = os.path.join(MODELS_DIR, "numerai_model.pt")
    scaler_path = os.path.join(MODELS_DIR, "numerai_scaler.pkl")
    
    if not os.path.exists(model_path):
        print("❌ No trained model. Run with --setup first.")
        return None
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    config = checkpoint["config"]
    features = checkpoint["features"]
    
    model = NumeraiNet(config["input_size"], config["hidden_sizes"], config["dropout"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Load live data
    print("📊 Loading live data...")
    live_df = pd.read_parquet(live_path)
    feature_cols = [c for c in live_df.columns if c.startswith("feature_")]
    
    # Use only features the model was trained on
    available = [f for f in features if f in feature_cols]
    X_live = live_df[available].values.astype(np.float32)
    
    # Handle NaN
    X_live = np.nan_to_num(X_live, nan=0.0)
    X_live = scaler.transform(X_live)
    
    # Predict
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_live).to(DEVICE)
        preds = model(X_tensor).cpu().numpy()
    
    # Create submission
    submission = pd.DataFrame({
        "id": live_df["id"],
        "prediction": preds
    })
    
    sub_path = os.path.join(NUMERAI_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"✅ Predictions saved: {sub_path} ({len(submission)} rows)")
    return sub_path


def submit_predictions(model_name="neural_net_v1"):
    """Submit predictions to Numerai tournament."""
    public_id = os.environ.get("NUMERAI_PUBLIC_ID")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")
    
    if not public_id or not secret_key:
        print("❌ Set NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY environment variables")
        print("   Get keys from: https://numer.ai/account")
        return
    
    sub_path = os.path.join(NUMERAI_DIR, "submission.csv")
    if not os.path.exists(sub_path):
        print("❌ No submission file. Run predictions first.")
        return
    
    napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
    
    # Get model id
    models = napi.get_models()
    if model_name not in models:
        print(f"⚠️ Model '{model_name}' not found in your account.")
        print(f"   Available models: {list(models.keys())}")
        print(f"   Create a model at https://numer.ai/models")
        return
    
    model_id = models[model_name]
    print(f"📤 Submitting predictions for '{model_name}'...")
    
    napi.upload_predictions(sub_path, model_id=model_id)
    print("✅ Predictions submitted!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Numerai Tournament Pipeline")
    parser.add_argument("--setup", action="store_true", help="Download data and train model")
    parser.add_argument("--train", action="store_true", help="Train/retrain model")
    parser.add_argument("--predict", action="store_true", help="Generate predictions on live data")
    parser.add_argument("--submit", action="store_true", help="Download live data, predict, and submit")
    parser.add_argument("--sample", type=float, default=0.1, help="Training data sample fraction (default 0.1)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    args = parser.parse_args()
    
    if args.setup:
        download_data()
        train_numerai_model(epochs=args.epochs, sample_frac=args.sample)
    elif args.train:
        train_numerai_model(epochs=args.epochs, sample_frac=args.sample)
    elif args.predict:
        generate_predictions()
    elif args.submit:
        download_data()  # Get latest live data
        generate_predictions()
        submit_predictions()
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python numerai_pipeline.py --setup          # First time")
        print("  python numerai_pipeline.py --submit         # Weekly submission")
