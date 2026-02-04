# ESN Path Prediction V2 Improvement Validation

Validation records for prediction accuracy improvements in path_prediction_v2.

## Improvement Approach

### Kalman Filter Hybrid

Combines ESN predictions with Kalman filter predictions using weighted averaging.

**Implementation:**
```python
class SimpleKalmanFilter:
    """Kalman Filter for 2D position tracking - State: [x, y, vx, vy]"""
    def __init__(self, dt=0.1, process_noise=0.1, measurement_noise=0.05):
        # State transition, measurement, covariance matrices

class KalmanHybridESNPredictor:
    """ESN + Kalman Filter hybrid"""
    # Weighting: (1 - kalman_weight) * esn_pred + kalman_weight * kalman_pred
    # kalman_weight = 0.3
```

**Result:** **+16.4%** improvement on ETH dataset

## ETH Dataset Validation Results

### V1 vs V2 Comparison

| Ped ID | V1 (ESN) | V2 (Kalman Hybrid) | Improvement |
|--------|----------|-------------------|-------------|
| 399 | 0.620m | 0.578m | +6.8% |
| 168 | 1.094m | 0.888m | +18.8% |
| 269 | 1.014m | 0.851m | +16.1% |
| 177 | 0.845m | 0.692m | +18.1% |
| 178 | 0.931m | 0.755m | +18.9% |
| **Average** | **0.901m** | **0.753m** | **+16.4%** |

### Visualization

![V1 vs V2 Comparison](images/eth_v1_v2_comparison.png)

## Other Approaches Tried

The following approaches showed limited or negative effect and were not adopted:

### 1. Reservoir Size Expansion
- V1: 25 units → V2: 50 units
- Result: Accuracy degradation (insufficient training data)

### 2. Spectral Radius Increase
- V1: 0.8-0.9 → V2: 0.88-0.95
- Result: Instability (poor compatibility with online learning)

### 3. Direction Change Detection Tuning
- Angle and speed threshold detection
- Result: Effective only for some patterns

## Conclusion

- **Kalman filter hybrid** is the most effective improvement
- Combination of ESN's short-term prediction and Kalman filter's smoothing effect is effective
- 16.4% improvement confirmed on real data (ETH dataset)

## File Structure

```
tools/
├── eth_esn_batch.py           # Batch evaluation script
├── eth_esn_visualizer.py      # Visualization script
├── eth_v1_v2_comparison.py    # V1 vs V2 comparison script
└── data/
    ├── students001_train.txt  # ETH dataset
    └── biwi_eth.txt           # BIWI dataset
```

## Reproduction

```bash
# V1 vs V2 comparison
python3 tools/eth_v1_v2_comparison.py --ped_ids 399 168 269 177 178

# Batch evaluation
python3 tools/eth_esn_batch.py --ped_ids 399 168 269

# Visualization
python3 tools/eth_esn_visualizer.py --ped_ids 399 168 269
```
