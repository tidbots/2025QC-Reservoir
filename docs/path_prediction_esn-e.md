# ESN Path Prediction Node Details

Python-based path prediction node using Echo State Networks (ESN).

**File:** `path_prediction/ros2_ws/src/esn_path_prediction/esn_path_prediction/esn_path_prediction.py`

## Architecture

**Framework:** ReservoirPy + scikit-learn + scipy

**Ensemble Model:** Multiple ESNs (default: 10) with averaged predictions.

**Pipeline:**
1. Input smoothing (Savitzky-Golay)
2. Online standardization (EWMA)
3. Multi-model ESN inference
4. Online adaptation/retraining
5. Clipping and stabilization

## Core Components

### OnlineStandardizer Class

Online z-score normalization using exponential weighted moving average (EWMA).

```python
class OnlineStandardizer:
    def __init__(self, mean, var, alpha=0.02):
        self.alpha = 0.02  # learning rate
```

### Diverse ESN Creation

```python
create_diverse_esns(n_models=5, base_units=25, seed=42, rls_forgetting=0.99)
```

**Per-ESN Randomization:**
- Reservoir units: base_units ± random(-5, 5)
- Spectral radius: 0.8-0.9
- Leak rate: 0.35-0.6
- Input scaling: 0.2-0.4
- Bias: -0.2 to +0.2

### Savitzky-Golay Smoothing

```python
savgol_win(win, window_length=9, polyorder=2)
```

Denoises leg position history before ESN input.

## ROS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `legs_topic` | string | `/hri/leg_finder/leg_pose` | Input leg position topic |
| `frame_id` | string | `base_footprint` | Output frame |
| `warmup` | int | 5 | Samples needed before ESN initialization |
| `window` | int | 20 | Max history buffer size |
| `future_horizon` | int | 20 | Steps to predict |
| `n_models` | int | 10 | Number of ESNs in ensemble |
| `leg_update_hz` | float | 10.0 | Max frequency of leg updates |
| `update_rate_hz` | float | 20.0 | Prediction publication rate |

## Internal Tuning Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `sg_window` | 9 | Savitzky-Golay window |
| `sg_poly` | 2 | SG polynomial order |
| `adapt_window` | 5 | Recent data window for adaptation |
| `sudden_change_thresh` | 0.6 | Distance to reset ESN state |
| `adapt_damping_nominal` | 0.35 | RLS learning rate (normal) |
| `adapt_damping_boost` | 1.0 | RLS learning rate (high error) |
| `boost_error_thresh` | 0.5 | Error threshold for boost mode |
| `state_clip` | 5.0 | Clipping limit for reservoir states |
| `wout_clip` | 8.0 | Clipping limit for readout weights |

## Workflow

### Phase 1: Warmup (0-5 samples)

```python
if len(self.history) >= max(self.warmup, 6):
    # Fit StandardScaler on history
    # Initialize OnlineStandardizers
    # Warm-start ESNs
```

### Phase 2: Leg Reception Callback

1. Frequency gating (≤10Hz)
2. Position extraction
3. History & buffer updates
4. Savitzky-Golay smoothing
5. Online standardizer update

### Phase 3: Prediction Loop (20Hz)

1. Sudden change detection (reset if > 0.6)
2. Multi-step rollout (20 steps per ESN)
3. Online adaptation (error-based boost mode)
4. Clipping and stabilization
5. Ensemble averaging

## Topics

### Published
| Topic | Type | Description |
|-------|------|-------------|
| `/hri/leg_finder/predicted_path` | nav_msgs/Path | 20-step future trajectory |

### Subscribed
| Topic | Type | Description |
|-------|------|-------------|
| `/hri/leg_finder/leg_pose` | PointStamped | Detected position from leg_finder |

## Tuning Guide

**For faster convergence:**
- `n_models`: Reduce 10 → 5
- `leg_update_hz`: Increase 10 → 20
- `adapt_damping_boost`: Increase 1.0 → 2.0

**For stability (noisy environments):**
- `sg_window`: Increase 9 → 11 or 13
- `sg_poly`: Decrease 2 → 1
- `adapt_damping_nominal`: Decrease 0.35 → 0.15

**For longer predictions:**
- `future_horizon`: Increase 20 → 30 or 40
- `window`: Increase 20 → 30
