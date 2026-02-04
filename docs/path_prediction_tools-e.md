# Visualization Tools

## ESN Visualizer

Standalone script to test and visualize the ESN path prediction algorithm without ROS.

**File:** `tools/esn_visualizer.py`

### Usage

```bash
# Test all patterns
python3 tools/esn_visualizer.py --pattern all --output output

# Specific pattern only
python3 tools/esn_visualizer.py --pattern straight --output output
```

### Test Patterns

| Pattern | Description | Characteristics |
|---------|-------------|-----------------|
| `straight` | Straight walking | Small lateral sway |
| `curve` | Curved walking | Arc trajectory |
| `zigzag` | Zigzag walking | Periodic lateral movement |
| `stop_and_go` | Stop and go | Includes stop periods |

### Output

Generates PNG images with 4 graphs for each pattern:

1. **Overall Trajectory and Predictions** - Actual trajectory (blue) and predictions (red)
2. **Prediction Detail** - Comparison of past, current, future, and prediction at specific time
3. **Prediction Error Over Time** - Error changes over time series
4. **Accuracy by Component** - Prediction accuracy per X/Y component

### Implementation

The script includes standalone implementations of:

- `SimpleESN` - Simple Echo State Network implementation
- `OnlineStandardizer` - Online Z-score normalization
- `ESNPredictor` - ESN predictor (ROS independent)

These reproduce the algorithms from `esn_path_prediction.py`.

### Example Results

**Prediction Accuracy (Mean Error):**

| Pattern | Mean Error | Std Dev |
|---------|------------|---------|
| straight | ~0.13 m | ~0.09 m |
| curve | ~0.21 m | ~0.18 m |
| zigzag | ~0.36 m | ~0.25 m |
| stop_and_go | ~0.18 m | ~0.18 m |

*Results vary due to random seed

### Customization

`ESNPredictor` class parameters can be modified:

```python
predictor = ESNPredictor(
    n_models=10,        # Number of ESNs
    warmup=5,           # Warmup samples
    window=20,          # History window size
    future_horizon=20   # Prediction steps
)
```

---

## ETH Dataset Tools

Evaluation and visualization tools using the ETH pedestrian tracking dataset.

See [ETH Dataset Evaluation](path_prediction_eth_evaluation-e.md) for details.

### Batch Evaluation (eth_esn_batch.py)

Evaluate ESN prediction accuracy without GUI.

```bash
# Default evaluation
python3 tools/eth_esn_batch.py

# Specify pedestrian IDs
python3 tools/eth_esn_batch.py --ped_ids 399 168 269
```

### Visualization (eth_esn_visualizer.py)

Visualize prediction results on ETH dataset.

```bash
python3 tools/eth_esn_visualizer.py --ped_ids 399 168 269
```

**Output:**
- Trajectory and predictions for each pedestrian
- Prediction comparisons at multiple frames
- Statistical summary

---

## Improvement Test Tool (esn_improvement_test.py)

Validate incremental improvement approaches.

```bash
python3 tools/esn_improvement_test.py --output output
```

**Validates:**
- Direction change detection tuning
- Kalman filter hybrid
- Combined approach

See [V2 Improvements](path_prediction_v2_improvements-e.md) for details.
