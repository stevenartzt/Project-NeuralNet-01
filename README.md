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

## RSI + Stochastic Strategy

Dedicated tools for testing RSI + Stochastic alignment:

```bash
# 1. Backtest the alignment strategy
python backtest_rsi_stoch_alignment.py --symbol SPY --days 1460

# 2. Run Monte Carlo simulation
python monte_carlo.py --symbol SPY --days 1460 --simulations 10000

# 3. Train neural net on RSI + Stochastic
python train_rsi_stochastic.py --symbol SPY --days 730 --epochs 100
```

### Strategy Rules
- **SHORT**: RSI > 70 AND Stochastic %K crosses below %D (both > 70)
- **LONG**: RSI < 30 AND Stochastic %K crosses above %D (both < 30)

### Custom Parameters
```bash
python backtest_rsi_stoch_alignment.py \
  --symbol SPY \
  --days 1460 \
  --rsi-high 75 \
  --rsi-low 25 \
  --stoch-high 80 \
  --stoch-low 20 \
  --hold 5 \
  --stop 2.0 \
  --target 3.0
```

### Monte Carlo Output
- Expected return distribution
- Risk of ruin (50% drawdown)
- Probability of profit
- Confidence intervals (5th, 25th, 75th, 95th percentiles)

## File Structure

```
neural-net/
├── fetch_data.py                    # Downloads OHLCV, computes 24 indicators
├── train.py                         # Neural net training engine
├── auto_optimize.py                 # Bayesian hyperparameter search
├── app.py                           # Flask dashboard with live visualizations
├── backtest_rsi_stoch_alignment.py  # RSI + Stochastic backtest
├── monte_carlo.py                   # Monte Carlo simulation
├── train_rsi_stochastic.py          # Neural net for RSI + Stochastic
├── requirements.txt                 # Python dependencies
├── data/                            # Training data (generated)
├── models/                          # Saved models & training history
├── training/results/                # Backtest & Monte Carlo results
└── optimize/                        # Optimization run results
```
