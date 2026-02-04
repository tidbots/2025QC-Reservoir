# ETH Dataset ESN Evaluation

Validation of ESN path prediction using the ETH pedestrian tracking dataset.

## Overview

The ETH dataset is a pedestrian tracking benchmark published by ETH Zurich. Using real pedestrian trajectory data enables more realistic evaluation compared to simulation patterns.

## Datasets

### students001_train.txt
- Location: ETH Campus
- Pedestrians: 400+
- Format: `frame  ped_id  x  y`
- Coordinates: meters

### biwi_eth.txt
- BIWI (Walking Pedestrians) dataset
- Same format

## Evaluation Tools

### Batch Evaluation (eth_esn_batch.py)

Evaluate ESN prediction accuracy without GUI.

```bash
# Default evaluation (5 pedestrians)
python3 tools/eth_esn_batch.py

# Specify pedestrian IDs
python3 tools/eth_esn_batch.py --ped_ids 399 168 269 177 178

# Parameter adjustment
python3 tools/eth_esn_batch.py --n_models 10 --future_horizon 20
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| --data | data/students001_train.txt | Dataset path |
| --ped_ids | Auto-select | Pedestrian IDs to evaluate |
| --n_peds | 5 | Number of pedestrians for auto-select |
| --n_models | 10 | Number of ESN models |
| --future_horizon | 20 | Prediction steps |

### Visualization (eth_esn_visualizer.py)

Visualize prediction results.

```bash
# Run visualization
python3 tools/eth_esn_visualizer.py --ped_ids 399 168 269

# Specify output directory
python3 tools/eth_esn_visualizer.py --output output
```

**Output:**
- Trajectory and predictions for each pedestrian
- Prediction comparisons at multiple frames
- Summary statistics

## Evaluation Results

### Test Conditions
- ESN models: 10
- Prediction horizon: 20 steps
- Warmup: 5 frames
- Window size: 20 frames

### Results Summary

| Ped ID | Mean Error (m) | Std Dev | Frames |
|--------|---------------|---------|--------|
| 399 | 0.620 | 0.346 | 301 |
| 168 | 1.094 | 1.146 | 202 |
| 269 | 1.014 | 0.508 | 191 |
| 177 | 0.845 | 0.857 | 184 |
| 178 | 0.931 | 0.543 | 184 |
| **Average** | **0.901** | | |

### Visualization Examples

#### Pedestrian 399 Predictions
![ETH Pedestrian 399](images/eth_ped_399.png)

#### Evaluation Summary
![ETH Summary](images/eth_summary.png)

## V1 vs V2 Comparison

### Comparison Tool (eth_v1_v2_comparison.py)

Compare V1 (original ESN) with V2 (Kalman hybrid) on ETH dataset.

```bash
python3 tools/eth_v1_v2_comparison.py --ped_ids 399 168 269 177 178
```

### Comparison Results

| Ped ID | V1 (ESN) | V2 (Kalman Hybrid) | Improvement |
|--------|----------|-------------------|-------------|
| 399 | 0.620m | 0.578m | +6.8% |
| 168 | 1.094m | 0.888m | +18.8% |
| 269 | 1.014m | 0.851m | +16.1% |
| 177 | 0.845m | 0.692m | +18.1% |
| 178 | 0.931m | 0.755m | +18.9% |
| **Average** | **0.901m** | **0.753m** | **+16.4%** |

### Comparison Visualization

![V1 vs V2 Comparison](images/eth_v1_v2_comparison.png)

### Conclusion

- V2 (Kalman hybrid) shows **16.4% average improvement** on real data
- Similar to simulation results (+18.8%)
- Improvement confirmed across all pedestrians

---

## Analysis

### Comparison with Simulation

| Evaluation Method | Mean Error | Prediction Horizon |
|-------------------|------------|-------------------|
| Simulation (synthetic patterns) | 0.14-0.27m | 20 steps |
| ETH Dataset | 0.62-1.09m | 20 steps |

Reasons for larger errors on ETH dataset:
1. **Coordinate scale**: ETH uses real-world meters (~15m range)
2. **Trajectory complexity**: Real pedestrians make unpredictable movements
3. **Speed variation**: Frequent stops, accelerations, direction changes

### ESN Applicability

- Longer trajectories (300+ frames) tend to have smaller errors
- Online learning adapts to trajectory patterns
- Sub-1m average error for 20-step prediction is practical

## File Structure

```
tools/
├── data/
│   ├── students001_train.txt  # ETH dataset
│   └── biwi_eth.txt           # BIWI dataset
├── eth_esn_batch.py           # Batch evaluation script
├── eth_esn_visualizer.py      # Visualization script
└── person_tracking_esn_fx.py  # Original script
```

## References

- ETH Walking Pedestrians Dataset: https://icu.ee.ethz.ch/research/datsets.html
- Pellegrini, S., et al. "You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking." ICCV 2009.
