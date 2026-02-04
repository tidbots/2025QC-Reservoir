# ETH Dataset ESN Evaluation

Validation of ESN path prediction using the ETH pedestrian tracking dataset.

## Overview

The ETH dataset is a pedestrian tracking benchmark published by ETH Zurich. Using real pedestrian trajectory data to evaluate ESN path prediction accuracy.

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

- V2 (Kalman hybrid) shows **16.4% average improvement**
- Improvement confirmed across all pedestrians
- Kalman filter smoothing effect is beneficial

---

## V3 Adaptive ESN

### Overview

V3 dynamically adjusts the weighting between ESN and Kalman filter based on trajectory complexity.

**Key Features:**
- **Trajectory Complexity Analysis**: Computes score from direction changes and speed variation
- **Adaptive Weight Adjustment**: Increases ESN weight for complex trajectories
- **Performance-based Feedback**: Weight adjustment based on recent prediction errors

### Comparison Tool (eth_v3_adaptive.py)

```bash
python3 tools/eth_v3_adaptive.py --ped_ids 399 168 269 177 178
```

### Comparison Results

| Ped ID | V1 (ESN) | V2 (Kalman Hybrid) | V3 (Adaptive) | vs V1 | vs V2 |
|--------|----------|-------------------|---------------|-------|-------|
| 399 | 0.620m | 0.578m | 0.453m | +26.9% | +21.6% |
| 168 | 1.094m | 0.888m | 0.742m | +32.2% | +16.4% |
| 269 | 1.014m | 0.851m | 0.698m | +31.2% | +18.0% |
| 177 | 0.845m | 0.692m | 0.589m | +30.3% | +14.9% |
| 178 | 0.931m | 0.755m | 0.635m | +31.8% | +15.9% |
| **Average** | **0.901m** | **0.753m** | **0.623m** | **+30.8%** | **+17.2%** |

### Visualization

![V3 Adaptive Comparison](images/eth_v3_adaptive.png)

### V3 vs Kalman Alone

| Method | Mean Error (m) | Notes |
|--------|---------------|-------|
| **Kalman alone** | **0.509** | Optimal for linear trajectories |
| V3 (Adaptive) | 0.623 | ESN+Kalman dynamic weighting |

**V3 is -22.4% worse than Kalman alone**

### Analysis

1. **V3 achieves 30.8% improvement over V1, 17.2% over V2**
2. **However, still underperforms Kalman alone**
   - ETH dataset contains mostly linear trajectories
   - Kalman filter's velocity-based prediction is optimal

---

## Complex vs Linear Trajectory Comparison

### Pedestrian Selection by Trajectory Complexity

Computed trajectory complexity scores within the ETH dataset to select and compare the most complex and most linear trajectories.

**Complexity Score Calculation:**
- Direction changes (average angle, maximum angle)
- Speed variation (coefficient of variation)
- Score = 0.6 × direction_score + 0.4 × speed_score

**Selected Pedestrians:**

| Group | Pedestrian IDs | Characteristics |
|-------|---------------|-----------------|
| Complex | 68, 90, 165, 399, 116 | Max angle >150°, sharp direction changes |
| Linear | 280, 248, 249, 273, 87 | Max angle <30°, mostly straight |

### Comparison Results

| Method | Complex (m) | Linear (m) |
|--------|-------------|------------|
| **Kalman** | **1.075** | **1.107** |
| V3 (Adaptive) | 1.523 | 1.188 |
| V2 (ESN+Kalman) | 1.903 | 1.309 |
| V1 (ESN) | 2.375 | 1.477 |
| Linear | 1.644 | 1.116 |

### V3 vs Kalman Gap

| Trajectory Type | Kalman | V3 | V3 Deficit |
|-----------------|--------|-----|------------|
| Complex | 1.075m | 1.523m | **-41.6%** |
| Linear | 1.107m | 1.188m | **-7.3%** |

### Analysis

1. **Kalman outperforms even on complex trajectories**
   - ETH's "complex" trajectories can still be handled by Kalman's velocity-based prediction
   - Direction changes are smooth with gradual velocity changes

2. **V3 approaches Kalman on linear trajectories**
   - V3's adaptive weighting shifts toward Kalman
   - Gap narrows to 7.3%

---

## Additional Verification: Cases Where ESN Excels

### A. BIWI Dataset Verification

Verification with BIWI dataset containing complex trajectories.

| Method | Mean Error (m) |
|--------|---------------|
| **Kalman** | **2.298** |
| Linear | 3.506 |
| V3 (Adaptive) | 3.875 |

→ Kalman is best even on BIWI

### B. Synthetic Data Verification

Verification with synthetic trajectories containing intentional sharp direction changes.

| Trajectory Type | Kalman | V3 | V2 | V1 | Best |
|-----------------|--------|-----|-----|-----|------|
| Linear | **0.071** | 0.153 | 0.217 | 0.312 | Kalman |
| Sharp 90° turn | **0.222** | 0.371 | 0.459 | 0.590 | Kalman |
| Zigzag | **0.593** | 0.672 | 0.736 | 0.857 | Kalman |
| Sudden stop→reverse | **0.293** | 0.521 | 0.603 | 0.773 | Kalman |

→ Kalman is best on all synthetic data

### D. Direction Change Segment Evaluation (Key Finding)

Extract and evaluate only direction change frames (angle change > 20°) from ETH dataset.

| Segment | Kalman | V3 | V2 | V1 | Best |
|---------|--------|-----|-----|-----|------|
| **Direction Change** | 3.107 | **2.924** | 2.964 | 3.088 | **V3** |
| Non-Direction-Change | **2.555** | 2.938 | 3.445 | 4.003 | **Kalman** |

**Key Finding:**
- **V3 outperforms Kalman by 5.9% during direction changes**
- Kalman is best during straight motion
- ESN's adaptive learning is effective for detecting and responding to direction changes

### Parameter Tuning

Grid search optimization of ESN parameters.

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| units | 25 | **50** | Larger reservoir capacity |
| spectral_radius | 0.8-0.9 | **0.90-0.98** | Improved long-term memory |
| leaking_rate | 0.35-0.6 | **0.50-0.70** | Faster adaptation |
| input_scaling | 0.2-0.4 | **0.40-0.60** | Higher input sensitivity |

**Tuning Results:**
- Baseline ESN error: 3.231m → Optimized: 3.072m (+4.9% improvement)
- Direction change improvement: 4.6% → **5.9%** (+1.3 points)

### Conclusion

1. **Kalman is best on overall average** - velocity-based prediction effective when straight motion dominates
2. **ESN is effective during direction changes** - adaptive learning responds to direction changes
3. **V3's adaptive weighting is valuable** - situation-dependent ESN/Kalman selection

### Evaluation Tools

```bash
# Synthetic data test
python3 tools/synthetic_trajectory_test.py --n_trials 3

# Direction change segment evaluation
python3 tools/eth_direction_change_test.py --angle_threshold 20 --n_peds 30
```

---

## Prediction Horizon Comparison

Verification with prediction steps extended from 20 to 50.

### Results

| Horizon | Kalman | ESN only | V3 (Hybrid) |
|---------|--------|----------|-------------|
| 20 | **1.93m** | 2.85m (-48%) | 2.21m (-15%) |
| 30 | **2.89m** | 4.30m (-49%) | 3.34m (-16%) |
| 40 | **3.91m** | 5.68m (-45%) | 4.48m (-15%) |
| 50 | **4.95m** | 7.23m (-46%) | 5.70m (-15%) |

### Analysis

- **Kalman maintains superiority even with extended horizon**
- ESN alone is about 45-50% worse than Kalman
- V3 (hybrid) is consistently about 15% worse

---

## Full ETH Dataset Evaluation (136 pedestrians)

V3 evaluation with optimized parameters on all data.

### Results

| Method | Mean Error (m) | vs V1 |
|--------|---------------|-------|
| **Kalman** | **1.184m** | - |
| V3 (Adaptive) | 1.316m | +24.9% |
| V2 (Fixed) | 1.504m | +14.2% |
| Linear | 1.539m | +12.2% |
| V1 (ESN) | 1.753m | - |

---

## Final Conclusions

### ETH Dataset Verification Results

1. **Kalman filter is most effective**
   - Pedestrian motion is essentially linear (approximately constant velocity)
   - Velocity-based prediction is optimal
   - Low computational cost

2. **ESN Limitations**
   - Online learning takes time to converge
   - Excessive complexity for linear motion
   - No improvement even with extended prediction horizon

3. **V3 (Adaptive Hybrid) Value**
   - 24.9% improvement over V1 (ESN alone)
   - Reduces gap with Kalman to about 15%
   - Adaptive weighting based on trajectory complexity is effective

### Conditions Where ESN May Be Effective

Not confirmed in this ETH dataset, but potentially effective under:

- More non-linear motion (robot control, complex time series)
- Data with repetitive patterns
- Time series with long-term dependencies
- Acceleration with QuantumCore library

---

## Non-linear Trajectory Test (Key Finding)

To test ESN effectiveness on more non-linear dynamics, synthetic trajectory data was generated.

### Test Tool

```bash
python3 tools/nonlinear_trajectory_test.py --n_trials 3 --future_horizon 20
```

### Trajectory Types Tested

| Type | Description |
|------|-------------|
| circle | Circular motion (periodic) |
| figure8 | Figure-8 (Lemniscate) |
| lissajous | Lissajous curve (3:2 ratio) |
| spiral | Expanding spiral |
| sinusoidal | Sinusoidal wave |
| double_sin | Double sinusoidal (two frequencies) |
| rosette | 5-petal rosette |
| lorenz_2d | Lorenz attractor (chaotic) |
| pendulum | Non-linear pendulum (large angle) |
| accelerate_decelerate | S-curve motion profile |

### Results: ESN Outperforms Kalman

| Trajectory | Kalman | V3 | V1 | Best | Improvement |
|------------|--------|-----|-----|------|-------------|
| **Lorenz 2D (Chaotic)** | 2.108 | 1.820 | **1.642** | V1 | **+22%** |
| **Non-linear Pendulum** | 14.087 | 11.796 | **10.706** | V1 | **+24%** |
| Double Sine | 2.947 | **2.811** | 3.005 | V3 | +4.6% |
| Rosette | 8.618 | **8.410** | 9.361 | V3 | +2.4% |

### Results: Kalman/Linear Better

| Trajectory | Kalman | V3 | Linear | Best |
|------------|--------|-----|--------|------|
| Circle | 0.871 | 1.229 | **0.569** | Linear |
| Sinusoidal | 1.274 | 1.489 | **0.826** | Linear |
| Accelerate/Decelerate | 0.873 | 1.053 | **0.622** | Linear |

### Key Conclusions

1. **Chaotic dynamics** (Lorenz attractor): ESN (V1) beats Kalman by **22%**
2. **Non-linear oscillatory systems** (large-angle pendulum): ESN (V1) beats Kalman by **24%**
3. **Periodic/linear motion**: Linear extrapolation or Kalman is optimal
4. **ESN's adaptive learning** is most effective when trajectory non-linearity is high

### Implications

- ESN is well-suited for **robot control trajectories** with non-linear dynamics
- **Chaotic systems** benefit from ESN's ability to learn complex patterns
- For **pedestrian prediction** (essentially linear), Kalman remains optimal

---

## Disturbance Response Test (Key Finding)

Evaluating prediction performance under sudden, unpredictable disturbances.

### Test Tool

```bash
python3 tools/disturbance_response_test.py --n_trials 3 --future_horizon 20
```

### Disturbance Types Tested

| Type | Description |
|------|-------------|
| sudden_obstacle | Sudden 90° turn due to obstacle |
| random_push | Random external force disturbances |
| goal_switch | Sudden goal/target change |
| sudden_stop_reverse | Stop and reverse direction |
| speed_burst | Sudden speed increases |
| erratic_motion | Smooth motion with erratic bursts |

### Results: During Disturbance Events

| Disturbance Type | Kalman | V3 | V1 | Best | Improvement |
|------------------|--------|-----|-----|------|-------------|
| **Random Push** | 3.471 | 3.037 | **2.892** | V1 | **+12.5%** |
| **Speed Burst** | 0.721 | **0.674** | 0.678 | V3 | **+6.6%** |
| Sudden Obstacle | **0.704** | 1.647 | 2.747 | Kalman | - |
| Goal Switch | **0.634** | 0.919 | 1.263 | Kalman | - |
| Sudden Stop/Reverse | **0.552** | 0.647 | 0.769 | Kalman | - |
| Erratic Motion | **1.000** | 1.046 | 1.209 | Kalman | - |

### Key Conclusions

| Disturbance Type | Characteristics | Best Method |
|------------------|-----------------|-------------|
| Continuous random forces | Unpredictable but continuous | **ESN** |
| Sudden but clear changes | Stable after change | **Kalman** |

**ESN Strength**: Online adaptation to continuous disturbances
**Kalman Strength**: Stable linear prediction after state changes

### Implications

- ESN excels when disturbances are **continuous and random** (e.g., wind, pushing)
- Kalman excels when disturbances cause **discrete state changes** (e.g., obstacle avoidance)
- For **robot control** with external forces, ESN may provide better prediction

---

## LSM (Liquid State Machine) Comparison (Key Finding)

Comparison between spiking neural network-based LSM and ESN.

### Test Tool

```bash
python3 tools/lsm_trajectory_test.py --n_trials 3 --future_horizon 20
```

### Results

| Trajectory | Kalman | ESN | LSM | Best | LSM vs ESN |
|------------|--------|-----|-----|------|------------|
| Linear | **0.041** | 0.067 | 0.051 | Kalman | +24.5% |
| Circle | 0.871 | 1.836 | 1.701 | Linear | +7.4% |
| **Lorenz (Chaotic)** | 2.109 | 1.790 | **1.581** | **LSM** | **+11.7%** |
| **Nonlinear Pendulum** | 14.085 | 18.665 | **9.734** | **LSM** | **+47.8%** |
| **Random Walk** | 0.788 | 1.155 | **0.775** | **LSM** | **+32.9%** |

### Key Findings

1. **LSM outperforms ESN in all cases** (especially 47.8% improvement on nonlinear pendulum)
2. **For non-linear dynamics**, LSM achieves best results (beats Kalman too)
3. **Spiking neurons** provide effective temporal information processing

### LSM Advantages

- Sparse, event-driven computation
- Better temporal pattern recognition through spike timing
- More biologically plausible dynamics
- Potentially more suitable for neuromorphic hardware

### ETH Dataset LSM Evaluation

| Method | Mean Error | vs Kalman |
|--------|-----------|-----------|
| **Kalman** | **0.971m** | - |
| LSM | 1.525m | -57.1% |
| ESN | 1.743m | -79.6% |

- On ETH (linear pedestrians), Kalman remains optimal
- **LSM improves over ESN by 12.5%**

### Implications

- For **chaotic and non-linear trajectories**, consider LSM over ESN
- For **linear motion**, Kalman is optimal; LSM still beats ESN
- LSM may be particularly effective for **robot control** applications
- Future work: Test LSM with QuantumCore acceleration

---

## Comparison with Conventional Methods

### Comparison Tool (eth_method_comparison.py)

Compare ESN with conventional trajectory prediction methods.

```bash
python3 tools/eth_method_comparison.py --ped_ids 399 168 269 177 178
```

### Compared Methods

| Method | Description | Source |
|--------|-------------|--------|
| Linear | Linear extrapolation | - |
| f(x) avg | Linear + Parabola + Sigmoid average | RSJ2025 1I5-03 |
| Kalman | Kalman filter only | - |
| ESN | Echo State Network ensemble | - |
| ESN+Kalman | ESN + Kalman hybrid | This project |

### Comparison Results

| Method | Mean Error (m) | vs Linear | vs Kalman |
|--------|---------------|----------|-----------|
| **Kalman** | **0.509** | **+25.6%** | - |
| V3 (Adaptive) | 0.623 | +8.9% | -22.4% |
| Linear | 0.684 | - | -34.4% |
| V2 (ESN+Kalman) | 0.753 | -10.1% | -47.9% |
| V1 (ESN) | 0.901 | -31.7% | -77.0% |
| f(x) avg | 3.443 | -403.5% | -576.4% |

### Comparison Visualization

![Method Comparison](images/eth_method_comparison.png)

### Analysis

1. **Kalman filter alone** achieves best results
   - Velocity-based prediction effective for linear pedestrian motion
   - Low computational cost

2. **f(x) average** is unstable
   - Sigmoid fitting diverges in some cases
   - Paper method effective under specific conditions but limited generality

3. **ESN+Kalman** improves over ESN alone
   - Kalman filter stability complements ESN prediction
   - ESN adaptive learning may help with complex trajectories

4. **ESN alone** can underperform conventional methods
   - Online learning needs time to converge
   - Possible overfitting on simple linear trajectories

---

## Analysis

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
├── eth_v1_v2_comparison.py    # V1 vs V2 comparison
├── eth_v3_adaptive.py         # V3 adaptive ESN evaluation
├── eth_method_comparison.py   # Conventional methods comparison
└── person_tracking_esn_fx.py  # Original script
```

## References

- ETH Walking Pedestrians Dataset: https://icu.ee.ethz.ch/research/datsets.html
- Pellegrini, S., et al. "You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking." ICCV 2009.
- Ono, Choi. "Pedestrian Avoidance by MPPI Control using 4WIDS Omnidirectional Mobile Robot." RSJ2025, 1I5-03. (Reference for f(x) prediction method)
