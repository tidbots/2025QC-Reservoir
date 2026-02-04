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

## Quick Reference

```bash
# Docker setup
cd path_prediction && ./docker_build.sh --user=roboworks --robot=hsrc30 --netif=<network-interface>
docker compose up

# Inside container
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch leg_finder leg_finder.launch.xml
ros2 run esn_path_prediction esn_path_prediction.py
```

## Key Packages

| Package | Language | Location |
|---------|----------|----------|
| leg_finder | C++ | `path_prediction/ros2_ws/src/leg_finder/` |
| esn_path_prediction | Python | `path_prediction/ros2_ws/src/esn_path_prediction/` |

## Detailed Documentation

See `docs/` for detailed documentation:
- `docs/architecture.md` / `docs/architecture-e.md` - System architecture
- `docs/setup.md` / `docs/setup-e.md` - Setup guide
