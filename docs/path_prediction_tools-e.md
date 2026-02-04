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
