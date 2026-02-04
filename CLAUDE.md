# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROS 2-based human-robot interaction system for detecting human legs via laser scanning and predicting their path using Echo State Networks (reservoir computing). Part of a NEDO initiative validating reservoir computing for service robot applications.

## Build & Run Commands

### Docker Setup (First Time)
```bash
cd path_prediction
./docker_build.sh --user=roboworks --robot=hsrc30 --netif=<network-interface>
docker compose up           # GPU-enabled
docker compose -f compose.no-gpu.yaml up  # CPU-only
```

### Inside Container
```bash
# Build ROS workspace
cd ~/ros2_ws && colcon build --symlink-install
source install/setup.bash

# Run leg detector
ros2 launch leg_finder leg_finder.launch.xml

# Enable detection
ros2 topic pub --once /hri/leg_finder/enable std_msgs/msg/Bool "data: true"

# Run path predictor
ros2 run esn_path_prediction esn_path_prediction.py
```

### Rebuild Docker Image
```bash
docker compose build
```

## Architecture

```
LaserScan (/scan)
    ↓
leg_finder (C++) - Geometric detection + Butterworth filtering
    ↓ /hri/leg_finder/leg_pose (PointStamped)
esn_path_prediction (Python) - ESN ensemble + online learning
    ↓ /hri/leg_finder/predicted_path (nav_msgs/Path)
```

### Key Packages

| Package | Language | Location | Purpose |
|---------|----------|----------|---------|
| leg_finder | C++ | `path_prediction/ros2_ws/src/leg_finder/` | Leg detection from laser scans |
| esn_path_prediction | Python | `path_prediction/ros2_ws/src/esn_path_prediction/` | Path prediction using ESN |

### ROS Topics

- **Input**: `/scan` (LaserScan)
- **Control**: `/hri/leg_finder/enable` (Bool), `/stop` (Empty)
- **Output**: `/hri/leg_finder/leg_pose` (PointStamped), `/hri/leg_finder/predicted_path` (Path)

## Technology Stack

- Ubuntu 22.04 + CUDA 11.7.1
- ROS 2 Humble with CycloneDDS
- ReservoirPy for Echo State Networks
- Docker with NVIDIA GPU support

## Key Configuration

- ROS Domain ID: 30 (configurable via `--domain`)
- CycloneDDS: Multicast disabled, peer-based discovery
- ESN: 10 ensemble models, 20-step prediction horizon
