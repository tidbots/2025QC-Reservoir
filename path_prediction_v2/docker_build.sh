#!/bin/bash

# Set default values
USER_NAME="roboworks"
GROUP_NAME="roboworks"
UID_VAL="1000"
GID_VAL="1000"
PASSWORD="tamagawa"
WORKSPACE_DIR=""
DOCKER_DIR=""
ROBOT_NAME="hsrc30"
NETWORK_IF="enp3s0"
ROS_IP="192.168.103.4"
DOMAIN_ID="30"

# Parse --key=value style arguments
for ARG in "$@"; do
  case $ARG in
    --user=*)
      USER_NAME="${ARG#*=}"
      ;;
    --group=*)
      GROUP_NAME="${ARG#*=}"
      ;;
    --uid=*)
      UID_VAL="${ARG#*=}"
      ;;
    --gid=*)
      GID_VAL="${ARG#*=}"
      ;;
    --password=*)
      PASSWORD="${ARG#*=}"
      ;;
    --workspace=*)
      WORKSPACE_DIR="${ARG#*=}"
      ;;
    --docker=*)
      DOCKER_DIR="${ARG#*=}"
      ;;
    --robot=*)
      ROBOT_NAME="${ARG#*=}"
      ;;
    --netif=*)
      NETWORK_IF="${ARG#*=}"
      ;;
    --rosip=*)
      ROS_IP="${ARG#*=}"
      ;;
    --domain=*)
      DOMAIN_ID="${ARG#*=}"
      ;;
    *)
      echo "Unknown option: $ARG"
      exit 1
      ;;
  esac
done

# Set WORKSPACE_DIR and DOCKER_DIR if not explicitly set
WORKSPACE_DIR=${WORKSPACE_DIR:-/home/$USER_NAME/share}
DOCKER_DIR=${DOCKER_DIR:-/home/$USER_NAME/docker/hsr_ros2}

# Output config summary
echo "Generating .env with:"
echo "USER_NAME=$USER_NAME"
echo "GROUP_NAME=$GROUP_NAME"
echo "UID=$UID_VAL"
echo "GID=$GID_VAL"
echo "PASSWORD=$PASSWORD"
echo "WORKSPACE_DIR=$WORKSPACE_DIR"
echo "DOCKER_DIR=$DOCKER_DIR"
echo "ROBOT_NAME=$ROBOT_NAME"
echo "NETWORK_IF=$NETWORK_IF"
echo "ROS_IP=$ROS_IP"
echo "DOMAIN_ID=$DOMAIN_ID"
echo ""

# Write to .env
cat <<EOF > .env
USER_NAME=$USER_NAME
GROUP_NAME=$GROUP_NAME
UID=$UID_VAL
GID=$GID_VAL
PASSWORD=$PASSWORD
WORKSPACE_DIR=$WORKSPACE_DIR
DOCKER_DIR=$DOCKER_DIR
ROBOT_NAME=$ROBOT_NAME
NETWORK_IF=$NETWORK_IF
ROS_IP=$ROS_IP
DOMAIN_ID=$DOMAIN_ID
EOF

# Run Docker Compose
docker compose build

