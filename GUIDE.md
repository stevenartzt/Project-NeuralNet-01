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