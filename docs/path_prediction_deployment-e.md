# Deployment Guide

Setting up and running the path_prediction system in a Docker environment.

## System Requirements

- Ubuntu 22.04
- Docker + Docker Compose
- NVIDIA GPU (optional, recommended)
- NVIDIA Container Toolkit (for GPU)

## File Structure

```
path_prediction/
├── Dockerfile                    # Container definition
├── compose.yaml                  # GPU-enabled compose
├── compose.no-gpu.yaml           # CPU-only compose
├── docker_build.sh               # Build helper
└── assets/
    ├── cyclonedds_profile.xml    # DDS network config
    ├── terminator_config         # Terminal config
    └── entrypoint.sh             # Container startup script
```

## Initial Setup

### 1. Run Build Script

```bash
cd path_prediction
./docker_build.sh --user=<username> --robot=<robot_name> --netif=<network_interface>
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--user` | roboworks | Username |
| `--uid` | 1000 | User ID |
| `--gid` | 1000 | Group ID |
| `--password` | tamagawa | Password |
| `--robot` | hsrc30 | Robot identifier |
| `--netif` | enp3s0 | Network interface |
| `--rosip` | 192.168.103.4 | ROS IP address |
| `--domain` | 30 | ROS Domain ID |

### 2. Build Docker Image

```bash
docker compose build
```

### 3. Start Container

```bash
# GPU enabled
docker compose up

# CPU only
docker compose -f compose.no-gpu.yaml up
```

## CycloneDDS Configuration

**File:** `assets/cyclonedds_profile.xml`

```xml
<Discovery>
  <Peers>
    <Peer Address="192.168.103.30"/>  <!-- Robot IP: UPDATE THIS -->
    <Peer Address="localhost"/>
  </Peers>
</Discovery>
```

**Important:** Update `<Peer Address>` to your actual robot/PC IP addresses.

### Configuration Steps

1. Identify robot IP on network
2. Update `<Peer Address>` in XML file
3. Set `--rosip` to container IP in `docker_build.sh`
4. Rebuild Docker image

## Docker Compose Configuration

### GPU Configuration (compose.yaml)

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          capabilities: [gpu]
```

### Network Configuration

```yaml
network_mode: "host"    # Required for DDS
ipc: host               # Shared memory
privileged: true        # Device access
```

### X11 Display Forwarding

```yaml
environment:
  - DISPLAY=${DISPLAY}
volumes:
  - /tmp/.X11-unix:/tmp/.X11-unix:rw
  - ~/.Xauthority:/home/${USER_NAME}/.Xauthority:ro
```

## Environment Variables

Key environment variables set inside container:

```bash
ROS_DOMAIN_ID=30
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
CYCLONEDDS_URI=/home/$USER_NAME/.config/cyclonedds_profile.xml
ROS_IP=<configured_value>
LANG=ja_JP.UTF-8
TZ=Asia/Tokyo
```

## Rebuilding

```bash
# Rebuild using existing .env file
docker compose build

# Full rebuild without cache
docker compose build --no-cache
```

## Troubleshooting

### Communication Issues

**Symptom:** No topics visible, nodes not communicating

**Solutions:**
1. Check `ROS_DOMAIN_ID`: `echo $ROS_DOMAIN_ID`
2. Verify CycloneDDS config has correct peer IPs
3. Check network interface: `ip a`
4. Test connectivity: `ping <robot_ip>`

### GPU Issues

**Symptom:** GPU not recognized

**Solutions:**
1. Install NVIDIA Container Toolkit
2. Verify with `nvidia-smi`
3. Use `compose.yaml` (not `compose.no-gpu.yaml`)

### Display Issues

**Symptom:** GUI apps don't display

**Solutions:**
```bash
xhost +local:docker
```
