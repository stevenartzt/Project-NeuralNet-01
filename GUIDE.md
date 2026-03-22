# Neural Networks: A Comprehensive Guide
## From Zero to Building Your Own — For Programmers

*This guide assumes programming experience but no machine learning background. We'll start from first principles and build up to understanding (and improving) the neural network in this codebase.*

---

# Part 1: Foundations (Basic)

## What is a Neural Network?

**The brain analogy (useful but imprecise):**
Your brain has ~86 billion neurons connected by synapses. When you see a cat, some neurons fire, pass signals to other neurons, and eventually you recognize "cat." Neural networks loosely mimic this — layers of artificial "neurons" passing numbers to each other.

**The precise definition:**
A neural network is a **function approximator** — a mathematical function that learns to map inputs to outputs by adjusting internal parameters (weights). Given enough parameters and training data, it can approximate any continuous function (this is the Universal Approximation Theorem).

```
Neural Network = Parameterized function f(x; θ) where θ = all weights and biases
Training = Finding θ that minimizes error between f(x) and desired output y
```

---

## A Single Neuron: The Building Block

A neuron does four things:
1. Takes inputs (x₁, x₂, ..., xₙ)
2. Multiplies each input by a weight (w₁, w₂, ..., wₙ)
3. Adds them up, plus a bias (b)
4. Passes the sum through an activation function (f)

**The math:**

```
z = (w₁ × x₁) + (w₂ × x₂) + ... + (wₙ × xₙ) + b
z = Σ(wᵢ × xᵢ) + b    (summation notation)
y = f(z)               (activation function)
```

**Visual:**

```
        ┌─────────────────────────────────────────────────────────┐
        │                      NEURON                             │
        │                                                         │
x₁ ──── │ ─── ×w₁ ───┐                                            │
        │            │                                            │
x₂ ──── │ ─── ×w₂ ───┼──→ [Σ + b] ──→ [f] ──→ y (output)         │
        │            │                                            │
x₃ ──── │ ─── ×w₃ ───┘                                            │
        │                                                         │
        └─────────────────────────────────────────────────────────┘

Where:
  Σ = sum all (weight × input) pairs
  b = bias (shifts the activation threshold)
  f = activation function (introduces non-linearity)
```

**In our code** (`train.py`, `DirectionNet` class):
Each `nn.Linear(prev_size, h)` creates a layer of neurons. The first layer might have 128 neurons, each receiving all 24 input features, each with its own 24 weights + 1 bias = 25 parameters per neuron = 3,200 parameters just in layer 1.

---

## Activation Functions

The activation function introduces **non-linearity**. Without it, stacking layers is pointless (see "Why Non-linearity Matters" below).

### ReLU (Rectified Linear Unit)
**Formula:** `f(x) = max(0, x)`

```
Output │
       │        ╱
       │       ╱
       │      ╱
  0 ───┼─────●──────── x
       │     │
       │     │ (negative inputs → 0)
```

- **What it does:** Pass positive values through unchanged, zero out negatives
- **When to use:** Default choice for hidden layers. Fast, works well, avoids vanishing gradient
- **Downside:** "Dead neurons" — if a neuron always outputs 0, it stops learning

**Our code uses ReLU:** See `nn.ReLU()` in the layer stack.

### Sigmoid
**Formula:** `f(x) = 1 / (1 + e^(-x))`

```
Output │
   1.0 │          ─────────────
       │        ╱
   0.5 │───────●
       │      ╱
   0.0 │─────────────
       └────────────────────── x
           -4  -2   0   2   4
```

- **What it does:** Squashes any value to range (0, 1)
- **When to use:** Output layer for binary classification (probability interpretation)
- **Downside:** Vanishing gradient at extremes, slow to train

### Tanh (Hyperbolic Tangent)
**Formula:** `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`

```
Output │
   1.0 │          ─────────────
       │        ╱
   0.0 │───────●
       │      ╱
  -1.0 │─────────────
       └────────────────────── x
           -4  -2   0   2   4
```

- **What it does:** Like sigmoid but outputs range (-1, 1)
- **When to use:** When you need centered outputs. Sometimes better than sigmoid
- **Downside:** Same vanishing gradient problem

### Softmax
**Formula:** `softmax(xᵢ) = e^(xᵢ) / Σ(e^(xⱼ))` for all j

- **What it does:** Converts a vector of numbers into probabilities (sum to 1.0)
- **When to use:** Output layer for multi-class classification

**Example:**
```
Input logits:  [2.0, 1.0, 0.1]
After softmax: [0.659, 0.242, 0.099]  (sum = 1.0)
Interpretation: 65.9% class A, 24.2% class B, 9.9% class C
```

**Our code:** The final `nn.Linear(prev_size, num_classes)` outputs 3 raw values (logits), and `nn.CrossEntropyLoss` applies softmax internally.

### Leaky ReLU / GELU
Variations that fix ReLU's "dead neuron" problem:
- **Leaky ReLU:** `f(x) = x if x > 0 else 0.01x` (small slope for negatives)
- **GELU:** Smooth approximation, used in transformers

**Our `auto_optimize.py`** tries all three: `["relu", "leaky_relu", "gelu"]`

---

## Why Non-linearity Matters

**The problem without activation functions:**
Linear transformations stack into a single linear transformation.

```
Layer 1: y₁ = W₁x + b₁
Layer 2: y₂ = W₂y₁ + b₂ = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)

This collapses to: y = Wx + b  (just another linear function!)
```

No matter how many layers you stack, you can only learn linear relationships. You could replace 100 layers with 1 layer and get the same result.

**The solution:**
Insert non-linear activation functions between layers. Now the network can learn curves, not just straight lines.

```
Layer 1: y₁ = ReLU(W₁x + b₁)
Layer 2: y₂ = ReLU(W₂y₁ + b₂)

This CANNOT be collapsed. Each layer adds new representational power.
```

---

## Loss Functions

The loss function measures **how wrong** the prediction is. Training minimizes this number.

### Mean Squared Error (MSE)
**Formula:** `MSE = (1/n) × Σ(yᵢ - ŷᵢ)²`

```
where:
  yᵢ  = true value
  ŷᵢ  = predicted value
  n   = number of samples
```

- **When to use:** Regression (predicting continuous values like price, temperature)
- **Interpretation:** Average of squared differences. Punishes big errors more than small ones (quadratic penalty)

**Example:**
```
True:      [100, 200, 150]
Predicted: [110, 180, 140]
Errors:    [ 10, -20,  10]
Squared:   [100, 400, 100]
MSE = (100 + 400 + 100) / 3 = 200
```

### Cross-Entropy Loss
**Formula (binary):** `BCE = -(1/n) × Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]`

**Formula (multi-class):** `CE = -(1/n) × Σ[yᵢ log(ŷᵢ)]` where y is one-hot encoded

- **When to use:** Classification (predicting categories)
- **Interpretation:** Measures the "surprise" of predictions. Confident wrong answers are penalized harshly.

**Example:**
```
True class: 0 (one-hot: [1, 0, 0])
Predicted:  [0.8, 0.1, 0.1]  (good prediction)
CE = -[1×log(0.8) + 0×log(0.1) + 0×log(0.1)]
CE = -log(0.8) = 0.22

True class: 0
Predicted:  [0.2, 0.7, 0.1]  (bad prediction)
CE = -log(0.2) = 1.61  (much higher loss!)
```

**Our code uses `nn.CrossEntropyLoss`** because we're classifying into BUY/SELL/NEUTRAL.

---

## Full Worked Example: A Tiny Network

Let's trace actual numbers through a network with:
- 2 inputs
- 1 hidden layer with 2 neurons (ReLU activation)
- 1 output (sigmoid for binary classification)

### Network Architecture

```
        INPUT           HIDDEN             OUTPUT
        LAYER           LAYER              LAYER

          x₁ ─────┬────→ h₁ ─────┬────→ o₁
                  │  ╲   │   ╲   │
                  │   ╲  │    ╲  │
                  │    ╲ │     ╲ │
          x₂ ─────┴────→ h₂ ─────┘

Weights:
  w₁₁ = 0.5  (x₁ → h₁)    w₁₃ = 0.3  (h₁ → o₁)
  w₁₂ = -0.3 (x₁ → h₂)    w₂₃ = 0.8  (h₂ → o₁)
  w₂₁ = 0.2  (x₂ → h₁)
  w₂₂ = 0.7  (x₂ → h₂)

Biases:
  b₁ = 0.1 (h₁)    b₃ = -0.2 (o₁)
  b₂ = -0.1 (h₂)
```

### Step 1: Input
```
x₁ = 0.8
x₂ = 0.4
```

### Step 2: Hidden Layer Computation
**Neuron h₁:**
```
z₁ = (w₁₁ × x₁) + (w₂₁ × x₂) + b₁
z₁ = (0.5 × 0.8) + (0.2 × 0.4) + 0.1
z₁ = 0.4 + 0.08 + 0.1 = 0.58

h₁ = ReLU(z₁) = ReLU(0.58) = 0.58  (positive, passes through)
```

**Neuron h₂:**
```
z₂ = (w₁₂ × x₁) + (w₂₂ × x₂) + b₂
z₂ = (-0.3 × 0.8) + (0.7 × 0.4) + (-0.1)
z₂ = -0.24 + 0.28 - 0.1 = -0.06

h₂ = ReLU(z₂) = ReLU(-0.06) = 0  (negative, zeroed out!)
```

### Step 3: Output Layer Computation
```
z₃ = (w₁₃ × h₁) + (w₂₃ × h₂) + b₃
z₃ = (0.3 × 0.58) + (0.8 × 0) + (-0.2)
z₃ = 0.174 + 0 - 0.2 = -0.026

o₁ = sigmoid(z₃) = 1 / (1 + e^0.026) = 0.4935
```

### Result
**Final output: 0.4935** (interpret as 49.35% probability of class 1)

If the true label was 1 (class 1), this is a mediocre prediction — nearly 50/50 when it should be confident. The loss would be:
```
BCE = -[1 × log(0.4935)] = -log(0.4935) = 0.706
```

Backpropagation would then adjust all 7 weights and 3 biases to push this prediction higher.

---

# Part 2: How Networks Learn (Intermediate)

## Forward Pass

The forward pass is what we just did in the worked example: data flows from input, through each layer, to output.

```
┌─────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌──────────┐
│  Input  │───→│ Layer 1│───→│ Layer 2│───→│ Layer 3│───→│  Output  │
│ x = [24]│    │ h = 128│    │ h = 64 │    │ h = 32 │    │ [3 probs]│
└─────────┘    └────────┘    └────────┘    └────────┘    └──────────┘
              
Each arrow = matrix multiply + bias + activation
```

**Our code** (`train.py`):
```python
def forward(self, x):
    return self.net(x)  # Sequential layers, PyTorch handles it
```

---

## Loss Calculation

After forward pass, compare prediction to truth:

```
Prediction: [0.3, 0.5, 0.2] (probabilities for SELL, NEUTRAL, BUY)
Truth:      [0, 1, 0]       (one-hot: actual class is NEUTRAL)

Cross-entropy loss = -log(0.5) = 0.693
```

**Goal:** Make this number smaller. A perfect prediction of [0, 1, 0] would give loss = -log(1.0) = 0.

---

## Backpropagation

Backpropagation answers: "How much did each weight contribute to the error?"

**The Chain Rule** (calculus):
If y = f(g(x)), then dy/dx = (dy/dg) × (dg/dx)

Applied to neural networks:
```
Loss depends on → output
Output depends on → layer 3 weights
Layer 3 depends on → layer 2 outputs → layer 2 weights → ...

∂Loss/∂w₁ = (∂Loss/∂output) × (∂output/∂layer3) × (∂layer3/∂layer2) × (∂layer2/∂w₁)
```

Gradients flow backward from loss to each weight:

```
┌─────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌──────┐
│  Input  │←───│ Layer 1│←───│ Layer 2│←───│ Layer 3│←───│ Loss │
│         │    │ ∂L/∂w₁ │    │ ∂L/∂w₂ │    │ ∂L/∂w₃ │    │      │
└─────────┘    └────────┘    └────────┘    └────────┘    └──────┘

← Gradients flow backward (backpropagation)
```

**Our code:**
```python
loss.backward()  # PyTorch computes all gradients automatically
```

---

## Gradient Descent

Once we know the gradients, update each weight:

```
new_weight = old_weight - (learning_rate × gradient)
```

**Intuition:** The gradient points "uphill" (direction of increasing loss). We want to go downhill (decreasing loss), so we subtract.

**Visual:**

```
Loss
  │    ╲
  │     ╲
  │      ╲       ╱
  │       ╲     ╱
  │        ╲   ╱
  │         ╲ ╱ ← We want to reach this minimum
  │          ●
  └────────────────── weight value
            
Gradient tells us: "go left to decrease loss"
We update: weight = weight - lr × gradient
```

---

## Learning Rate

**Too high:** Overshoot the minimum, bounce around, never converge

```
Loss
  │ ←─────→ ←─────→ ←─────→  (bouncing!)
  │    ╲       ╱ ╲     ╱
  │     ╲     ╱   ╲   ╱
  │      ╲   ╱     ╲ ╱
  │       ● ←───────● 
  └────────────────────
```

**Too low:** Converge painfully slowly, might get stuck in local minima

```
Loss
  │
  │  ╲               
  │   ╲              
  │    ●●●●●●●●●●●●●●●●●●●●●●●●●●●●● (tiny steps, takes forever)
  │                            ↘
  └────────────────────────────────
```

**Just right:** Steady progress toward minimum

**Typical values:** 0.001 (1e-3) is a common starting point. Our code uses `lr=0.001`.

---

## Batches, Epochs, Iterations

**Batch:** A subset of training data processed together before updating weights.
```
Total data: 22,593 samples
Batch size: 256
→ 88 batches per epoch (22,593 / 256 ≈ 88)
```

**Iteration:** One batch processed = one weight update.

**Epoch:** One complete pass through all training data.
```
1 epoch = 88 iterations (with batch size 256)
100 epochs = 8,800 iterations
```

**Why batches?**
- **Batch size 1** (SGD): Noisy updates, slow, but escapes local minima
- **Full batch:** Smooth updates, fast computation, but might converge to sharp minima
- **Mini-batch (32-512):** Best of both worlds. Our code uses `batch_size=256`.

---

## Optimization Algorithms

### SGD (Stochastic Gradient Descent)
The basic algorithm: update weights using gradient of a random mini-batch.

```
w = w - lr × gradient
```

**Problem:** Can oscillate in ravines (long, narrow valleys in loss landscape).

### Momentum
Add "velocity" — accumulate past gradients.

```
v = momentum × v_prev + gradient
w = w - lr × v
```

**Like a ball rolling downhill:** It builds up speed and can roll through small bumps. Typical momentum = 0.9.

### Adam (Adaptive Moment Estimation)
Combines momentum with per-parameter adaptive learning rates.

```
m = β₁ × m_prev + (1 - β₁) × gradient           (momentum)
v = β₂ × v_prev + (1 - β₂) × gradient²          (squared gradient)
m_hat = m / (1 - β₁^t)                          (bias correction)
v_hat = v / (1 - β₂^t)
w = w - lr × m_hat / (√v_hat + ε)               (update)
```

**Translation:** Features that rarely have gradients get larger updates. Features that frequently have gradients get smaller updates. Self-adjusting.

**Our code uses Adam:**
```python
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
```

---

## Backprop Math: A Small Example

Let's compute gradients for our tiny network from Part 1.

**Setup:**
- Input: x₁=0.8, x₂=0.4
- Hidden neuron output: h₁=0.58, h₂=0
- Final output: o=0.4935
- True label: y=1
- Loss (BCE): L = -log(0.4935) = 0.706

**Step 1: Gradient at output**
```
∂L/∂o = -y/o + (1-y)/(1-o)
      = -1/0.4935 + 0
      = -2.026
```

**Step 2: Through sigmoid**
```
∂o/∂z₃ = o × (1-o) = 0.4935 × 0.5065 = 0.250

∂L/∂z₃ = ∂L/∂o × ∂o/∂z₃ = -2.026 × 0.250 = -0.507
```

**Step 3: Gradients for output layer weights**
```
∂L/∂w₁₃ = ∂L/∂z₃ × h₁ = -0.507 × 0.58 = -0.294
∂L/∂w₂₃ = ∂L/∂z₃ × h₂ = -0.507 × 0 = 0
∂L/∂b₃  = ∂L/∂z₃ × 1  = -0.507
```

**Step 4: Backprop through hidden layer** (continue chain rule)
```
∂L/∂h₁ = ∂L/∂z₃ × w₁₃ = -0.507 × 0.3 = -0.152
∂L/∂h₂ = ∂L/∂z₃ × w₂₃ = -0.507 × 0.8 = -0.406

∂L/∂z₁ = ∂L/∂h₁ × ReLU'(z₁) = -0.152 × 1 = -0.152  (z₁ was positive)
∂L/∂z₂ = ∂L/∂h₂ × ReLU'(z₂) = -0.406 × 0 = 0       (z₂ was negative → ReLU gradient is 0)
```

**Step 5: Gradients for hidden layer weights**
```
∂L/∂w₁₁ = ∂L/∂z₁ × x₁ = -0.152 × 0.8 = -0.122
∂L/∂w₂₁ = ∂L/∂z₁ × x₂ = -0.152 × 0.4 = -0.061
```

**Update example (lr=0.1):**
```
w₁₁_new = w₁₁ - lr × ∂L/∂w₁₁
        = 0.5 - 0.1 × (-0.122)
        = 0.5 + 0.0122 = 0.5122
```

The gradient was negative (loss decreases when w₁₁ increases), so we increased the weight.

---

# Part 3: Architecture & Design (Intermediate-Advanced)

## Layer Types

### Dense / Fully-Connected
Every neuron connects to every neuron in the previous layer.

```
Input:  [24 features]
Dense:  [128 neurons]
Each neuron has 24 weights → 24 × 128 = 3,072 parameters (+ 128 biases)
```

**Use for:** Tabular data, final classification layers.

**Our entire network is dense** — appropriate for tabular stock data.

### Convolutional (CNN)
Neurons only connect to a local region ("receptive field"). Share weights across spatial positions.

```
Image input: 28×28 pixels
Filter: 3×3 (9 weights, shared across the image)
Slides across the image, detecting edges/patterns locally
```

**Use for:** Images, spatial data. NOT what we use here.

### Recurrent (RNN/LSTM)
Connections form cycles — output feeds back as input for the next timestep.

```
x₁ → [RNN] → h₁
       ↓
x₂ → [RNN] → h₂   (h₁ is passed to next step)
       ↓
x₃ → [RNN] → h₃
       ↓
      output
```

**Use for:** Sequences, time series, text. **This is what we should use for stock data** (see Part 6).

---

## Depth vs Width Tradeoffs

**Wider networks** (more neurons per layer):
- Learn more features at each level
- Easier to train (gradients don't vanish as much)
- Risk: overfitting

**Deeper networks** (more layers):
- Learn more abstract/hierarchical features
- Can represent more complex functions
- Risk: vanishing gradients, harder to train

**Rules of thumb:**
- Start with 2-3 hidden layers
- First layer often largest, then shrink (funnel shape)
- Total parameters ≈ 10× training samples is a loose upper bound

**Our architecture:** `[128, 64, 32]` — a funnel. With 24 inputs and 3 outputs:
```
Layer 1: 24 × 128 + 128 = 3,200 parameters
Layer 2: 128 × 64 + 64  = 8,256 parameters
Layer 3: 64 × 32 + 32   = 2,080 parameters
Output:  32 × 3 + 3     = 99 parameters
Total: ~13,600 parameters
```

---

## Batch Normalization

**The problem:** As training progresses, the distribution of inputs to each layer shifts ("internal covariate shift"). Networks have to constantly adapt to changing distributions.

**The solution:** Normalize each layer's inputs to have mean=0, variance=1.

**Formula:**
```
x_norm = (x - mean) / sqrt(variance + ε)
output = γ × x_norm + β    (learnable scale and shift)
```

**Benefits:**
- Faster training (can use higher learning rates)
- Acts as regularization
- Reduces sensitivity to initialization

**Our code:**
```python
nn.BatchNorm1d(h),  # After each linear layer, before activation
```

---

## Dropout

**The problem:** Networks memorize training data (overfitting).

**The solution:** During training, randomly "drop" (zero out) some neurons.

```
Training:
[0.5, 0.3, 0.8, 0.2] → (drop 30%) → [0.5, 0, 0.8, 0]

Inference:
Use all neurons, but scale by (1 - dropout_rate) to compensate
```

**Why it works:** Forces the network to learn redundant representations. No neuron can rely on specific other neurons being present.

**Our code:**
```python
nn.Dropout(dropout),  # dropout=0.3 means 30% dropped
```

**Typical values:** 0.2-0.5 for hidden layers, lower for input layer.

---

## Regularization: L1 and L2

Regularization adds a penalty to the loss function for having large weights.

### L2 Regularization (Weight Decay)
**Formula:** `Loss_total = Loss_data + λ × Σ(w²)`

Penalizes large weights. Pushes weights toward zero but doesn't force them to exactly zero.

**Effect:** Smoother models, less overfitting.

### L1 Regularization
**Formula:** `Loss_total = Loss_data + λ × Σ|w|`

Also penalizes large weights, but **creates sparse models** — many weights become exactly zero.

**Effect:** Feature selection (irrelevant features get zero weight).

**Our code uses L2 (weight decay):**
```python
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
```

---

## Architecture Decisions: Rules of Thumb

**How many layers?**
- Start with 2-3 hidden layers
- Add more only if underfitting and you have lots of data
- Beyond 5-6 layers, consider skip connections (ResNet-style)

**How many neurons per layer?**
- First hidden: 2-4× the number of input features
- Subsequent layers: decrease by factor of 2 (funnel shape)
- Or keep constant if you have lots of data

**Our `auto_optimize.py` searches:**
```python
n_layers = trial.suggest_int("n_layers", 1, 4)
hidden_sizes = [trial.suggest_int(f"hidden_{i}", 16, 128, step=16) for i in range(n_layers)]
```

---

# Part 4: Training in Practice (Advanced)

## Train/Test/Validation Splits

```
All Data
├── Training Set (60-80%): Learn weights
├── Validation Set (10-20%): Tune hyperparameters, early stopping
└── Test Set (10-20%): Final evaluation, never touch during training
```

**Why validation?**
If you tune hyperparameters on the test set, you're effectively "training" on it. The test set must be held out completely.

**Our code** splits 80/20 (train/test), using the last 20% for testing:
```python
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
```

**Important for time series:** Always split by time, not randomly. You can't train on future data and test on past data (data leakage).

---

## Overfitting vs Underfitting

```
         Training Error    Test Error     Status
High     High              High           Underfitting (model too simple)
Low      Low               Low            Good fit
Low      Low               HIGH           Overfitting (model too complex)
```

**Visual (Bias-Variance Tradeoff):**

```
Error │
      │   Test error                ╱
      │             ╲             ╱
      │              ╲   ╱───────╱
      │               ╲ ╱
      │                ●  Sweet spot
      │               ╱
      │   Train error╱
      └─────────────────────────────────
                   Model complexity →
        Underfitting │   │ Overfitting
```

---

## Early Stopping

Stop training when validation loss stops improving.

```python
if test_acc > best_test_acc:
    best_test_acc = test_acc
    patience_counter = 0
    save_model()  # Save best model
else:
    patience_counter += 1
    if patience_counter >= patience:
        break  # Stop training
```

**Our code uses patience=15** — stop if no improvement for 15 epochs.

---

## Learning Rate Scheduling

Reduce learning rate as training progresses.

### Step Decay
Multiply LR by factor every N epochs.
```
Epochs 1-30:  lr = 0.001
Epochs 31-60: lr = 0.0005
Epochs 61-90: lr = 0.00025
```

### Cosine Annealing
Smoothly decrease LR following a cosine curve.
```
lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t / T))
```

### ReduceLROnPlateau (What we use)
Reduce LR when a metric stops improving.

**Our code:**
```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
# If loss doesn't improve for 5 epochs, multiply LR by 0.5
```

---

## Hyperparameter Tuning

### Grid Search
Try every combination.
```
lr:       [0.001, 0.01, 0.1]
dropout:  [0.2, 0.3, 0.4]
layers:   [[128], [128, 64], [128, 64, 32]]

Total: 3 × 3 × 3 = 27 combinations
```
**Pros:** Thorough. **Cons:** Exponential explosion, expensive.

### Random Search
Randomly sample combinations. Often finds good results faster than grid search.

### Bayesian Optimization (Optuna) — What we use

Intelligently explores the search space:
1. Train a few random configurations
2. Build a probabilistic model of which regions are promising
3. Sample next configuration from promising regions
4. Update model, repeat

**Our `auto_optimize.py` uses Optuna:**
```python
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
)
study.optimize(objective, n_trials=200)
```

**Pruning:** Optuna can stop bad trials early. If a trial is performing below median after 3 epochs, kill it.

---

## Data Preprocessing

### Standardization (Z-score normalization)
```
x_scaled = (x - mean) / std

Result: mean=0, std=1
```
**Use for:** Most neural networks (what we use).

### Min-Max Normalization
```
x_scaled = (x - min) / (max - min)

Result: range [0, 1]
```
**Use for:** When you need bounded values.

**Our code:**
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on train only!
X_test = scaler.transform(X_test)        # Apply same transform to test
```

**Critical:** Fit the scaler on training data only. If you fit on all data, you leak test information.

---

## Class Imbalance

When some classes are much more common than others.

**Our data:**
```
SELL:    7,234 samples
NEUTRAL: 8,891 samples
BUY:     6,468 samples
```
Fairly balanced, but not perfect.

### Solutions:

**Weighted Loss:** Penalize mistakes on rare classes more.
```python
class_weights = 1.0 / class_counts  # Inverse frequency
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
```

**Oversampling:** Duplicate minority class samples.

**SMOTE:** Generate synthetic minority samples by interpolating between existing ones.

**Our code uses weighted loss** — see `train.py`:
```python
class_weights = 1.0 / (class_counts + 1)
class_weights = class_weights / class_weights.sum() * 3
```

---

## Cross-Validation

Instead of a single train/test split, train multiple times on different splits.

### K-Fold Cross-Validation
```
Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]

Final score = average of 5 folds
```

### Time-Series Split (for sequential data)
```
Fold 1: [Train] [Test]
Fold 2: [Train] [Train] [Test]
Fold 3: [Train] [Train] [Train] [Test]
```
Never train on future data.

**We use time-ordered split** (last 20% is test) but not full k-fold CV — it's expensive for neural nets.

---

# Part 5: Types of Neural Networks (Advanced)

## MLP (Multi-Layer Perceptron) — What We're Using

```
Input → Dense → Dense → Dense → Output
        ↓       ↓       ↓
     [ReLU]  [ReLU]  [Softmax]
```

**Good for:**
- Tabular data (spreadsheet-like)
- When features are independent
- When order doesn't matter

**Our use case:** 24 technical indicators → 3 classes (BUY/SELL/NEUTRAL)

**Limitations:** Treats each row independently. Doesn't know that yesterday's RSI relates to today's RSI.

---

## CNN (Convolutional Neural Network)

```
Image → [Conv] → [Pool] → [Conv] → [Pool] → [Dense] → Output
        detect   reduce    detect   reduce   classify
        edges    size      shapes   size
```

**Key idea:** Local connectivity + weight sharing. A 3×3 filter slides across the image, detecting patterns anywhere they appear.

**Good for:**
- Images
- Any data with spatial structure
- Time series (1D convolution)

**Example architecture (image classification):**
```
Input: 224×224×3 (RGB image)
Conv1: 64 filters, 3×3 → 224×224×64
Pool1: 2×2 max pooling → 112×112×64
Conv2: 128 filters, 3×3 → 112×112×128
Pool2: 2×2 max pooling → 56×56×128
...
Flatten → Dense → Softmax → 1000 classes
```

---

## RNN / LSTM (Recurrent Neural Networks)

```
Time step 1:  x₁ → [RNN] → h₁ (hidden state)
                     ↓
Time step 2:  x₂ → [RNN] → h₂ (receives h₁)
                     ↓
Time step 3:  x₃ → [RNN] → h₃ (receives h₂)
                     ↓
                  Output
```

**Key idea:** Memory. The hidden state carries information from previous timesteps.

**LSTM (Long Short-Term Memory):** Solves the vanishing gradient problem in RNNs with "gates" that control information flow.

```
┌─────────────────────────────────────────────────────┐
│                    LSTM Cell                         │
│                                                      │
│  [Forget Gate] — what to discard from memory        │
│  [Input Gate]  — what new information to store      │
│  [Output Gate] — what to output from memory         │
│                                                      │
│  Cell State (long-term memory) ─────────────────→   │
│  Hidden State (short-term memory) ───────────────→  │
└─────────────────────────────────────────────────────┘
```

**Good for:**
- Time series (stock prices!)
- Sequences (text, audio)
- Variable-length inputs

**Why we should use this:** Stock data IS a time series. Today's RSI of 65 means something different if yesterday was 30 (recovering) vs 80 (declining). An LSTM would capture this.

---

## Transformers

**Key idea:** Attention mechanism — let every element in a sequence attend to every other element.

```
"The cat sat on the mat"
   ↓    ↓   ↓  ↓   ↓   ↓
   Each word attends to all other words
   "cat" attends strongly to "sat" and "mat"
```

**Architecture:**
```
Input → [Embedding] → [Self-Attention] → [Feed-Forward] → ... → Output
                           ↑
                     Multi-head attention:
                     Q (query), K (key), V (value)
                     Attention(Q,K,V) = softmax(QK^T / √d) × V
```

**Why transformers dominate:**
- Parallelizable (unlike RNNs)
- Can attend to any position (long-range dependencies)
- Scale incredibly well with data and parameters

**GPT, BERT, Claude:** All transformer-based.

**For stock data:** Transformers could capture long-range patterns (what happened 20 days ago might matter), but need lots of data.

---

## GANs (Generative Adversarial Networks)

Two networks fighting each other:

```
Generator: Random noise → Fake images
                ↓
Discriminator: Real or Fake? ← Real images
                ↓
Generator improves to fool Discriminator
Discriminator improves to catch Generator
```

**Good for:**
- Image generation
- Data augmentation
- Style transfer

---

## Autoencoders

Compress data, then reconstruct it.

```
Input → [Encoder] → Compressed representation → [Decoder] → Reconstruction
 784        128                10                  128          784
(image)                    (bottleneck)                      (image)
```

**Good for:**
- Dimensionality reduction
- Anomaly detection (bad reconstructions = anomalies)
- Feature learning

---

## When to Use Each Type

| Data Type | Network Type | Why |
|-----------|--------------|-----|
| Tabular (spreadsheet) | MLP | No spatial/temporal structure |
| Images | CNN | Spatial patterns, local features |
| Time series | LSTM/Transformer | Sequential dependencies |
| Text | Transformer | Long-range dependencies, attention |
| Generation | GAN/VAE | Adversarial or probabilistic generation |
| Anomaly detection | Autoencoder | Learn "normal," flag outliers |

**Our situation:** We're using MLP on what is really time-series data. This is a fundamental limitation.

---

# Part 6: Our System — Current Limitations & Next Steps

## Current Architecture (train.py)

```
┌────────────────────────────────────────────────────────────────────┐
│                         DirectionNet                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Input: 24 technical indicators                                   │
│     ↓                                                              │
│  [Linear 24→128] → [BatchNorm] → [ReLU] → [Dropout 0.3]          │
│     ↓                                                              │
│  [Linear 128→64] → [BatchNorm] → [ReLU] → [Dropout 0.3]          │
│     ↓                                                              │
│  [Linear 64→32]  → [BatchNorm] → [ReLU] → [Dropout 0.3]          │
│     ↓                                                              │
│  [Linear 32→3] → (Softmax via CrossEntropyLoss)                   │
│     ↓                                                              │
│  Output: [P(SELL), P(NEUTRAL), P(BUY)]                            │
│                                                                    │
│  Total parameters: ~13,600                                        │
│  Training samples: 22,593 across 50 tickers                       │
│  Test accuracy: ~40% (random baseline: 33%)                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Current Performance

```
Model           | Accuracy | vs Random
----------------|----------|----------
Random Baseline | 33%      | —
Neural Net      | 40%      | +7%
Random Forest   | 41%      | +8%
XGBoost         | 39%      | +6%
LightGBM        | 40%      | +7%
```

**Observation:** Tree-based models roughly match the neural net. This suggests the bottleneck is NOT the model architecture — it's the data and features.

---

## Limitation 1: Not Enough Data

**Problem:** 22,593 samples across 50 tickers. Deep learning typically needs 100K+ samples to really shine.

**Why it matters:**
- With limited data, neural nets overfit easily
- Can't learn complex patterns reliably
- Tree models are more data-efficient

**Fix:** Add more tickers, use longer history, include intraday data.

```python
# Current (fetch_data.py)
TICKERS = [...50 tickers...]
period = "2y"

# Better
TICKERS = [...200+ tickers...]
period = "5y"
# Or switch to hourly data: interval="1h"
```

---

## Limitation 2: Daily Data Only

**Problem:** We're using daily bars. By the time we see today's close, the opportunity may be gone.

**Missing patterns:**
- Intraday momentum (first hour predicts the day?)
- Overnight gaps
- Volume patterns within the day

**Fix:** Use hourly data (yfinance supports `interval="1h"` for last 730 days).

---

## Limitation 3: No Sequence Modeling

**Problem:** We treat each data point independently. The model sees today's RSI=65 but doesn't know if it was 30 yesterday (recovering) or 80 (declining).

```
Current: Each row is independent
─────────────────────────────────────────────
Day 1: RSI=30, MACD=-2, ... → PREDICT
Day 2: RSI=45, MACD=-1, ... → PREDICT (no memory of Day 1)
Day 3: RSI=65, MACD=0.5, ... → PREDICT (no memory of Days 1-2)
```

**Fix:** Use LSTM or Transformer.

```python
# Instead of DirectionNet (MLP), use:
class DirectionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        # Use last timestep's output
        return self.fc(lstm_out[:, -1, :])
```

**Data reshaping needed:** Instead of single rows, create sequences.
```
Input to LSTM: last 20 days of indicators → predict tomorrow
Shape: (batch_size, 20, 24)
```

---

## Limitation 4: No Market Regime Awareness

**Problem:** The model doesn't know if we're in a bull market, bear market, or sideways chop. The same RSI=30 means different things in different regimes.

**Fix:** Add market regime features:
```python
# In fetch_data.py, add:
df["spy_trend"] = spy_sma50 > spy_sma200  # Bull/bear
df["vix_level"] = vix_close               # Fear gauge
df["spy_momentum"] = spy_return_20d       # Market momentum
```

---

## Limitation 5: No Cross-Asset Features

**Problem:** Stocks don't move in isolation. If SPY drops 2%, most stocks follow. We ignore this.

**Fix:** Include market-wide features:
```python
# Fetch SPY, VIX, TLT (bonds) alongside each ticker
# Add as features:
"spy_return_1d", "vix_level", "vix_change", "tlt_return_1d"
```

---

## Limitation 6: Limited Feature Set

**What we have:** 24 technical indicators (RSI, MACD, Bollinger, etc.)

**What we're missing:**
- Volume profile (price levels with high volume)
- Options flow (unusual activity suggests informed trading)
- Sentiment (news, social media)
- Sector relative strength
- Earnings proximity

---

## Limitation 7: Threshold Tuning

**Current:** BUY if future_return > 0.5%, SELL if < -0.5%

**Problem:** Is 0.5% the right threshold?
- Too tight: More signals, but noisier
- Too loose: Fewer signals, but more confident

**Fix:** Try different thresholds or make it a hyperparameter:
```python
threshold = trial.suggest_float("threshold", 0.003, 0.015)
```

---

## Limitation 8: Single-Day Target

**Current:** Predict tomorrow's direction.

**Problem:** Day-to-day noise is high. Might have better signal on longer horizons.

**Fix:** Try multiple horizons:
```python
df["label_1d"] = create_labels(df, forward_days=1)
df["label_3d"] = create_labels(df, forward_days=3)
df["label_5d"] = create_labels(df, forward_days=5)
```

---

## Recommended Next Steps (Priority Order)

### 1. Add Sequence Modeling (LSTM)
**Effort:** Medium | **Impact:** High

This is the biggest architectural gap. Stock data is sequential — we should treat it that way.

### 2. Add Market Regime Features
**Effort:** Low | **Impact:** Medium

Quick win. Fetch SPY/VIX alongside tickers, add as features.

### 3. Scale Up Data
**Effort:** Low | **Impact:** Medium

Add more tickers (200+), extend history to 5 years.

### 4. Try Hourly Data
**Effort:** Medium | **Impact:** High

Intraday patterns might be stronger. More data points per ticker.

### 5. Multi-Horizon Targets
**Effort:** Low | **Impact:** Medium

Train separate models for 1-day, 3-day, 5-day predictions. See which has more signal.

### 6. Ensemble Methods
**Effort:** Medium | **Impact:** Medium

Combine neural net predictions with tree model predictions. Often beats either alone.

---

## Code Reference

| File | Purpose |
|------|---------|
| `fetch_data.py` | Downloads OHLCV, computes 24 technical indicators, creates labels |
| `train.py` | Defines DirectionNet, training loop, saves history for dashboard |
| `auto_optimize.py` | Bayesian hyperparameter search with Optuna, compares to tree baselines |
| `app.py` | Streamlit dashboard for visualization |

---

## Summary: The Path Forward

```
Current State (40% accuracy, +7% vs random):
┌──────────────────────────────────────────────────────────────────┐
│  MLP on daily data, no sequence modeling, limited features      │
└──────────────────────────────────────────────────────────────────┘

Target State (aim for 50%+ accuracy):
┌──────────────────────────────────────────────────────────────────┐
│  LSTM on hourly data, market regime features, 200+ tickers,     │
│  multiple horizons, ensemble with tree models                    │
└──────────────────────────────────────────────────────────────────┘
```

The neural network architecture is sound. The bottleneck is:
1. **Data structure:** Not using sequences
2. **Data quantity:** Need more samples
3. **Data quality:** Missing important features (market regime, cross-asset)
4. **Signal-to-noise:** Daily moves are noisy; intraday or longer horizons might help

The tree models matching neural net performance is actually informative — it tells us the issue isn't model expressiveness, it's the features and data representation.

---

*Guide created for the neural-net project. See the live dashboard at `python -m streamlit run app.py`.*