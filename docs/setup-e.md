# Setup Guide

## Docker Setup (First Time)

```bash
cd path_prediction
./docker_build.sh --user=roboworks --robot=hsrc30 --netif=<network-interface>
docker compose up           # GPU-enabled
docker compose -f compose.no-gpu.yaml up  # CPU-only
```

## Inside Container

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

## Rebuild Docker Image

```bash
docker compose build
```
