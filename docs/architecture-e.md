# Architecture

## System Configuration

```
LaserScan (/scan)
    ↓
leg_finder (C++) - Geometric detection + Butterworth filtering
    ↓ /hri/leg_finder/leg_pose (PointStamped)
esn_path_prediction (Python) - ESN ensemble + online learning
    ↓ /hri/leg_finder/predicted_path (nav_msgs/Path)
```

## Key Packages

| Package | Language | Location | Purpose |
|---------|----------|----------|---------|
| leg_finder | C++ | `path_prediction/ros2_ws/src/leg_finder/` | Leg detection from laser scans |
| esn_path_prediction | Python | `path_prediction/ros2_ws/src/esn_path_prediction/` | Path prediction using ESN |

## ROS Topics

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
