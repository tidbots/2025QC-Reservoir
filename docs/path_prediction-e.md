# Path Prediction System Overview

ROS 2-based human-robot interaction (HRI) system for detecting human legs using LIDAR and predicting their path using Echo State Networks (ESN).

## System Architecture

```
LIDAR Scan (/scan)
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

### Input
- `/scan` (sensor_msgs/LaserScan) - LIDAR scan data

### Control
- `/hri/leg_finder/enable` (std_msgs/Bool) - Enable/disable detection
- `/stop` (std_msgs/Empty) - Emergency stop

### Output
- `/hri/leg_finder/leg_pose` (geometry_msgs/PointStamped) - Detected leg position
- `/hri/leg_finder/legs_found` (std_msgs/Bool) - Leg detection status
- `/hri/leg_finder/predicted_path` (nav_msgs/Path) - Predicted path (20 steps ahead)
- `/hri/leg_finder/hypothesis` (visualization_msgs/Marker) - Debug visualization

## Technology Stack

- Ubuntu 22.04 + CUDA 11.7.1
- ROS 2 Humble + CycloneDDS
- ReservoirPy (Echo State Networks)
- Docker + NVIDIA GPU support

## Quick Start

```bash
# Docker setup
cd path_prediction
./docker_build.sh --user=roboworks --robot=hsrc30 --netif=<network-interface>
docker compose up

# Inside container
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash

# Terminal 1: Leg detector
ros2 launch leg_finder leg_finder.launch.xml

# Terminal 2: Enable detection
ros2 topic pub --once /hri/leg_finder/enable std_msgs/msg/Bool "data: true"

# Terminal 3: Path prediction
ros2 run esn_path_prediction esn_path_prediction.py
```

## Detailed Documentation

- [leg_finder Details](path_prediction_leg_finder-e.md) - Leg detection algorithm
- [ESN Path Prediction Details](path_prediction_esn-e.md) - ESN algorithm
- [Deployment](path_prediction_deployment-e.md) - Docker configuration
