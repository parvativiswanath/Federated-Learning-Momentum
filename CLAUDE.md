# FedWAN – Codebase Guide

## Project Overview

This is an implementation of **FedWAN** (Federated Learning with Weighted Averaging and Nesterov momentum), a FL algorithm that combines:
- **Nesterov Accelerated Gradient (NAG)** for client-side local training
- **KL-divergence-based weighted aggregation** on the server, prioritizing clients with more diverse data distributions

The paper shows 3–12% convergence improvement on Non-IID data vs existing FL solutions.

---

## File Structure

```
federated.py          # Main entry point – full FL training loop + all aggregation logic
model/
  layers.py           # Model definitions: CNN, DNN, LogisticRegression
  train.py            # Client training functions: plain SGD, momentum, NAG, Mime
dataset/
  data_utils.py       # Dataset loading (MNIST, CIFAR-10, EMNIST) + data split strategies
merger.py             # Post-run utility to combine per-algorithm CSVs into one file
requirements.txt      # Dependencies
```

---

## Algorithms Implemented

All algorithms are implemented from scratch (no external FL libraries).

| Algorithm | Client Training | Server Aggregation |
|-----------|----------------|-------------------|
| **FedAvg** | Plain SGD (`train`) | Simple average (`fed_avg`) |
| **MFL** | Momentum SGD (`train_with_momentum`) | Average models + average velocities (`fed_momentum_nag`) |
| **FedNAG** | NAG (`train_with_NAG`) | Average models + average velocities (`fed_momentum_nag`) |
| **FedWAN** *(proposed)* | NAG (`train_with_NAG`) | KL-weighted model avg + average velocities (`fed_wan`) |
| **FedMom** | Plain SGD (`train`) | Server-side momentum on pseudo-gradient (`fed_mom`) |
| **Mime** | SGD + global velocity correction (`train_mime`) | Average models + update server velocity (`mime`) |

---

## Models (`model/layers.py`)

- **LogisticRegression**: Linear(784→10) + log_softmax. MNIST only.
- **DNN**: 3-layer MLP (inputdim²×3 → 64 → 32 → 10) + log_softmax. Designed for CIFAR-10 (inputdim=32).
- **CNN**: 2 conv layers + 2 FC layers + log_softmax. Designed for MNIST (hardcoded 320-dim flatten).

**Known issue**: `inputdim` in `layers.py` is currently set to 32 (CIFAR-10). The CNN's `fc1 = Linear(320, 50)` is hardcoded for MNIST's 28×28 input (after two 2×2 max-pools on 1-channel input). Switching datasets requires manual changes in `layers.py` and `federated.py`.

---

## Training Functions (`model/train.py`)

All training functions use:
- `epochs = 3`, `learningRate = 0.001`, `momentum = 0.9`
- Loss: `nn.NLLLoss()` (paired with `log_softmax` output)
- Manual parameter updates (not `optimizer.step()`) for momentum/NAG

### `train(model, dataset)`
Plain SGD. Used by FedAvg and FedMom clients.

### `train_with_momentum(model, dataset, velocity)`
Manual SGD+momentum. Velocity update (PyTorch convention):
```
velocity = momentum * velocity + grad
param -= lr * velocity
```

### `train_with_NAG(model, dataset, velocity)`
Nesterov update:
```
velocity = momentum * velocity - lr * grad
param += velocity
param += momentum * velocity   # lookahead step
```
**Note**: The NAG implementation applies two `param.add_` calls per step — a standard velocity step followed by an extra lookahead. Review carefully if modifying.

### `train_mime(model, dataset, global_velocity)`
Applies global server velocity as a correction term during local training:
```
param -= lr * (momentum * global_velocity + grad)
```
Then computes a single-batch full-gradient on the *server model* for the Mime server velocity update.

---

## Data Splitting (`dataset/mnist_dataset.py`)

Three dataset split strategies exist:

| Function | Type | Used? |
|----------|------|-------|
| `split_client_datasets` | IID, round-based | Commented out |
| `split_client_datasets_non_iid` | Non-IID, random size per round | Commented out |
| `test` | Non-IID, random class distribution (MNIST) | Commented out |
| `test2` | Non-IID, class-count-proportional heterogeneity | **Active** |
| `split_non_iid_client_datasets` | Non-IID, explicit class distribution | Available |

**Active split (`split_non_iid_class_proportional`)**: Client `i+1` (in reversed order) is assigned `2*(i+1)` classes as "primary" classes plus 50 random samples from all other classes. Each client's dataset is capped at 8,000 samples.

---

## `federated.py` – Main Logic

### Global State
```python
clientModels        # list of trained client models per round
clientVelocities    # list of client velocities (for momentum algorithms)
clientDistributions # label count dicts per client (for FedWAN)
client_data_sizes   # number of samples per client
clientGrads         # full-batch gradients per client (for Mime)
clientModelsLock    # threading lock protecting above lists
```

### Configuration
- `clientNum = 5`, `trainingRounds = 20`, `momentum = 0.9` — global constants
- `federatedConfig` class inside `federated()` has **separate** values: 5 clients, **15 rounds** — this overrides the global `trainingRounds`

### `fedWAN` Aggregation — Key Details and Known Issues
1. **Global distribution**: hardcoded as `800` per label instead of summing actual client counts (line ~169). This approximation was intentional for the paper results but is not the "correct" global distribution.
2. **KL divergence**: uses raw counts (not probabilities) for `p`; `q` is `count/total_size` (normalized). The formula is non-standard.
3. Weights are `1/KL_i`, then normalized to sum to 1. Clients more similar to the global distribution get lower weight.

### `fedmom` — Server-Side Momentum
Computes pseudo-gradient as `global_params - client_state` (sum over clients), then applies momentum on server parameters. The `globalVelocity` parameter passed in is never read — a fresh `global_velocity` is initialized to zeros each round.

### Threading
Clients train in parallel using `threading.Thread`. The lock `clientModelsLock` protects appends to the shared lists.

### CSV Output
Each run writes a file `federated_metrics_cnn{algo}_test1.csv`. The header has 3 columns (`time`, `accuracy`, `loss`) but each row writes 4 values (round+1, curr_time, accuracy, loss) — the round column is off by one in the header.

---

## Running the Code

Edit the bottom of `federated.py`:
```python
if __name__ == "__main__":
    federated('fedwan')   # or 'fedavg', 'fednag', 'mfl', 'mime', 'fedmom'
```

To switch datasets/models, manually change:
1. `trainSet`/`testSet` in `federated.py` (comment/uncomment)
2. `serverModel = LogisticRegression()` in `federated()` → swap with `CNN()` or `DNN()`
3. `inputdim` in `layers.py` if switching between MNIST (28) and CIFAR-10 (32)

---

## Known Issues / Warts

1. **Model/filename mismatch**: filename is `cnn{algo}` but `serverModel` is hardcoded to `LogisticRegression()`.
2. **`fedmom` velocity bug**: `globalVelocity` argument is unused; server momentum resets each round (intentionally preserved).
3. **Mime full-batch gradient**: only uses a single batch from the iterator (`next(iter(dataset))`), not the true full batch.
