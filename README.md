# Neural Net — Direction Engine

Independent neural network for predicting stock direction from raw technical indicators.
No dependency on any external scoring system — learns entirely from price data.

## Architecture

**Network 1 — Direction Predictor**
- Input: 24 technical indicators computed from raw OHLCV data
- Output: BUY / SELL / NEUTRAL with confidence probabilities
- Architecture: Flexible MLP (auto-optimized via Bayesian search)

**Auto-Optimizer**
- Bayesian hyperparameter search (Optuna)
- Tests hundreds of architectures automatically
- Aggressive pruning kills bad trials early
- Tree-based baselines (XGBoost, LightGBM, Random Forest) for comparison

## Setup

```bash
# Clone
git clone https://github.com/stevenartzt/neural-net.git
cd neural-net

# Install dependencies
pip install -r requirements.txt

# For GPU (CUDA) — much faster training:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Usage

```bash
# 1. Fetch training data (2 years of S&P 500 daily OHLCV)
python fetch_data.py

# 2. Train a model
python train.py

# 3. Run auto-optimizer (finds best architecture)
python auto_optimize.py 200    # 200 trials

# 4. Launch dashboard
python app.py
# Open http://localhost:5005
```

## Dashboard Features

- **Loss & Accuracy curves** — live per epoch
- **Network topology** — animated neuron activations
- **Confusion matrix** — BUY/SELL/NEUTRAL predictions vs actual
- **Feature importance** — what the net learned matters (emergent)
- **Live inference** — watch single predictions flow through the network
- **Auto-optimizer** — Bayesian hyperparameter search with live progress
- **Tree baselines** — instant comparison vs XGBoost/LightGBM/RF

## GPU Training

If you have an NVIDIA GPU with CUDA, training will automatically use it.
The auto-optimizer on a 4070 Ti should complete 200 trials in ~2-3 minutes
vs ~10 minutes on CPU.

To verify GPU is available:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your GPU
```

## File Structure

```
neural-net/
├── fetch_data.py       # Downloads OHLCV, computes 24 indicators
├── train.py            # Neural net training engine
├── auto_optimize.py    # Bayesian hyperparameter search
├── app.py              # Flask dashboard with live visualizations
├── requirements.txt    # Python dependencies
├── data/               # Training data (generated)
├── models/             # Saved models & training history
└── optimize/           # Optimization run results
```
