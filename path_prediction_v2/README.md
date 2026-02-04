# ROS 2 Leg Finder + Path Prediction (Dockerized)

This repository contains a **ROS 2 project running inside Docker** for **human–robot interaction (HRI)**. It provides:

- A **leg detection node** that publishes the detected human leg position
- A **path prediction node** (ESN-based) that predicts the future path from leg motion
- A **Docker + Docker Compose workflow** to ensure reproducible builds and networking

The system is intended to run on or alongside a robot, with explicit configuration for **CycloneDDS**, **ROS_DOMAIN_ID**, and network interfaces.

---

## Repository Structure

```
.
├── assets/
│   └── cyclonedds_profile.xml   # DDS configuration (IP must be updated)
├── docker_build.sh              # First-time build + environment setup script
├── compose.yml
├── compose.no-gpu.yml
├── .env                         # Auto-generated (DO NOT COMMIT)
├── ros2_ws/src/leg_finder/
├── ros2_ws/src/esn_path_prediction/
└── README.md
```

---

## Prerequisites

- Docker
- Docker Compose (v2)
- ROS 2-compatible robot or PC on the same network
- CycloneDDS (used as DDS middleware)

> **Note**: This project assumes you are familiar with basic ROS 2 commands (`ros2 launch`, `ros2 run`, `ros2 topic`).

---

## 1. Configure CycloneDDS (IMPORTANT)

Before building the container, you **must update the IP address** inside the CycloneDDS profile.

1. Open the file:
   ```
   assets/cyclonedds_profile.xml
   ```

2. Update the IP-related field (e.g., `<NetworkInterfaceAddress>` or equivalent) to match the **robot’s network IP**.

This ensures correct DDS discovery and communication across machines.

---

## 2. First-Time Build (Environment Setup)

The first build must be done using the provided helper script. This script:

- Generates a local `.env` file
- Sets user, group, network, ROS, and robot-specific variables
- Builds the Docker images using Docker Compose

### Default Build

```bash
./docker_build.sh
```

### Custom Build Options

You may override any parameter using `--key=value` arguments:

```bash
./docker_build.sh \
  --user=roboworks \
  --uid=1000 \
  --gid=1000 \
  --robot=hsrc30 \
  --netif=enp3s0 \
  --rosip=192.168.103.4 \
  --domain=30
```

### Generated `.env` Variables

The script automatically creates a `.env` file containing:

- `USER_NAME`, `GROUP_NAME`
- `UID`, `GID`, `PASSWORD`
- `WORKSPACE_DIR`, `DOCKER_DIR`
- `ROBOT_NAME`
- `NETWORK_IF`
- `ROS_IP`
- `DOMAIN_ID`

> ⚠️ The `.env` file is **local-only** and should not be committed to Git.

---

## 3. Rebuilding After First Setup

Once the `.env` file exists, you can rebuild normally using:

```bash
docker compose build
```

Use `docker_build.sh` again **only if you need to change environment variables**.

---

## 4. Run the System

Start the containers with:

```bash
docker compose up
```

A **Terminator terminal** will open automatically.

---

## Configuring the Laser Scan Input

The **leg_finder** node is sensor-agnostic and can work with **any `sensor_msgs/LaserScan` topic**.

This is configured via arguments in the launch file:

- `laser_scan_topic` (default: `/scan`)
- `laser_scan_frame` (default: `base_range_sensor_link`)

These arguments are forwarded as parameters to the `leg_finder_node`.

### Option A – Edit the Launch File

You may directly edit:

```
leg_finder/launch/leg_finder.launch.xml
```

And update the default values:

```
<arg name="laser_scan_topic" default="/your/laser/topic"/>
<arg name="laser_scan_frame" default="your_laser_frame"/>
```

### Option B – Override via Command Line (Recommended)

You can override the laser scan configuration at runtime without modifying the file:

```bash
ros2 launch leg_finder leg_finder.launch.xml \
  laser_scan_topic:=/your/laser/topic \
  laser_scan_frame:=your_laser_frame
```

This allows the leg finder to be reused across different robots and sensors.

---

## 5. Runtime Execution (Inside the Container)

### Terminal 1 – Launch Leg Finder

```bash
ros2 launch leg_finder leg_finder.launch.xml
```

This node publishes detected leg positions as:

```
/hri/leg_finder/leg_pose
```

Message type:

```
geometry_msgs/msg/PointStamped
```

QoS:

```
rclcpp::SensorDataQoS()
```

---

### Terminal 2 – Enable the Leg Detector

The leg detector is disabled by default. Enable it by publishing:

```bash
ros2 topic pub --once \
  /hri/leg_finder/enable \
  std_msgs/msg/Bool \
  data:\ false
```

---

### Terminal 3 – Run the Path Predictor

```bash
ros2 run esn_path_prediction esn_path_prediction.py
```

#### Subscribed Topic

```
/hri/leg_finder/leg_pose
```

```python
self.sub_ = self.create_subscription(
    PointStamped,
    self.legs_topic,
    self.cb_leg,
    rclpy.qos.qos_profile_sensor_data
)
```

#### Published Topic

```
/hri/leg_finder/predicted_path
```

```python
self.pub_path_ = self.create_publisher(
    Path,
    '/hri/leg_finder/predicted_path',
    1
)
```

---

## Data Flow Overview

```
Leg Finder
   ↓  /hri/leg_finder/leg_pose
Path Predictor (ESN)
   ↓  /hri/leg_finder/predicted_path
```

---

## Notes & Tips

- Make sure `ROS_DOMAIN_ID` matches across all machines
- Verify the network interface (`NETWORK_IF`) matches your system
- CycloneDDS misconfiguration is the most common source of communication issues

---

## License

[Add license information here]

---

## Maintainer

Luis Contreras

