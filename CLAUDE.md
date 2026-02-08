# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Policy

- All documentation must be saved in `docs/` directory in Markdown format
- Always create both Japanese and English versions:
  - Japanese: `filename.md`
  - English: `filename-e.md`
- `README.md` contains overview only (Japanese)
- `README-e.md` is the English version of README
- Detailed documentation goes in `docs/`

## Project Overview

ROS 2-based human-robot interaction system for detecting human legs via laser scanning and predicting their path using Echo State Networks (reservoir computing). Part of a NEDO initiative validating reservoir computing for service robot applications.

## Architecture

### ROS 2 Data Pipeline

```
LIDAR (/scan) → leg_finder (C++) → /hri/leg_finder/leg_pose (PointStamped)
    → esn_path_prediction (Python) → /hri/leg_finder/predicted_path (nav_msgs/Path, 20 steps)
```

### System Variants

There are two deployable ROS 2 versions and standalone evaluation tools:

- **`path_prediction/`** — V1: Pure ESN ensemble (original). Git submodule from `gitlab.com/tidbots/path_prediction.git`
- **`path_prediction_v2/`** — V2: ESN + Kalman hybrid with fixed blending weights
- **`tools/`** — Standalone evaluation scripts (no ROS dependency). These implement V1, V2, V3 (adaptive), LSM, and baselines for offline comparison against trajectory datasets

### Algorithm Variants (in tools/)

| Variant | Script | Description |
|---------|--------|-------------|
| V1 (ESN) | `eth_method_comparison.py` | ESN ensemble with online learning |
| V2 (Hybrid) | `eth_v1_v2_comparison.py` | ESN + Kalman with fixed weights |
| V3 (Adaptive) | `eth_v3_adaptive.py` | Dynamic complexity-based ESN/Kalman weighting |
| LSM | `lsm_trajectory_test.py` | Liquid State Machine with LIF neurons |
| Baselines | various | Linear extrapolation, Kalman filter, f(x) average |

### Key Libraries

- **reservoirpy** — ESN implementation (`Reservoir`, `RLS` nodes)
- **scipy** — Savitzky-Golay filtering, curve fitting
- **sklearn** — `StandardScaler`, `Ridge` regression for LSM readout

### Shared Patterns Across Scripts

- `OnlineStandardizer` class — Exponential moving average normalization for real-time use (duplicated in each script, not shared)
- Kalman filter — Implemented inline (constant velocity model) in comparison scripts
- All tool scripts use `matplotlib.use('Agg')` for headless rendering and save to `output/`
- ETH dataset loaded from `tools/data/students001_train.txt` (columns: frame, ped_id, y, x)
- Scripts accept `--output-dir` argument for output path

## Quick Reference

```bash
# Docker setup (V1)
cd path_prediction && ./docker_build.sh --user=roboworks --robot=hsrc30 --netif=<network-interface>
docker compose up

# Inside container
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch leg_finder leg_finder.launch.xml
ros2 run esn_path_prediction esn_path_prediction.py

# Run evaluation tools (no ROS needed, run on host)
python3 tools/eth_method_comparison.py
python3 tools/lsm_trajectory_test.py
python3 tools/nonlinear_trajectory_test.py
python3 tools/disturbance_response_test.py

# Generate PDF docs from markdown
python3 tools/md2pdf.py
```

## Key Packages

| Package | Language | Location |
|---------|----------|----------|
| leg_finder | C++ | `path_prediction/ros2_ws/src/leg_finder/` |
| esn_path_prediction | Python | `path_prediction/ros2_ws/src/esn_path_prediction/` |

## ROS 2 Topics

| Topic | Type | Direction | Package |
|-------|------|-----------|---------|
| `/scan` | sensor_msgs/LaserScan | Input | leg_finder |
| `/hri/leg_finder/leg_pose` | geometry_msgs/PointStamped | Output/Input | leg_finder → esn |
| `/hri/leg_finder/predicted_path` | nav_msgs/Path | Output | esn_path_prediction |
| `/hri/leg_finder/enable` | std_msgs/Bool | Control | leg_finder |

## Evaluation Datasets

- **ETH (BIWI)** — `tools/data/students001_train.txt`, `biwi_eth.txt` (real pedestrian trajectories, linear)
- **UCY** — `tools/data/ucy_hotel.txt`, `ucy_zara01.txt`, `ucy_zara02.txt`, `ucy_students003.txt`
- **Synthetic** — Generated in-script (Lorenz attractor, non-linear pendulum, sinusoidal, figure-8)

## Detailed Documentation

See `docs/` for detailed documentation:
- `docs/path_prediction.md` / `docs/path_prediction-e.md` - System overview
- `docs/path_prediction_leg_finder.md` / `docs/path_prediction_leg_finder-e.md` - Leg detection
- `docs/path_prediction_esn.md` / `docs/path_prediction_esn-e.md` - ESN algorithm
- `docs/path_prediction_deployment.md` / `docs/path_prediction_deployment-e.md` - Docker setup
- `docs/path_prediction_tools.md` / `docs/path_prediction_tools-e.md` - Tool descriptions
- `docs/path_prediction_eth_evaluation.md` / `docs/path_prediction_eth_evaluation-e.md` - ETH evaluation results
