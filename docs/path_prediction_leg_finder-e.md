# leg_finder Node Details

Real-time 2D LIDAR-based leg detection node in C++.

**File:** `path_prediction/ros2_ws/src/leg_finder/src/leg_finder_node.cpp`

## Detection Algorithm

### Stage 1: Laser Range Filtering

```cpp
filter_laser_ranges()
```

- FILTER_THRESHOLD = 0.081m for noise removal
- Applies 2-point or 3-point moving average
- Zeros isolated or noisy points

### Stage 2: Downsampling (Optional)

```cpp
downsample_scan()
```

- Controlled by `scan_downsampling` parameter (default: 1)
- Skips scan points to reduce computational load

### Stage 3: Leg Hypothesis Detection

```cpp
find_leg_hypothesis()
```

1. Convert polar (range, angle) → Cartesian (x, y)
2. TF2 transform from laser frame to base_link frame
3. Flank detection (rapid range change > 0.04m)
4. Validate leg candidates with geometric criteria

**Leg Size Constraints:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| LEG_THIN | 0.00341m² | Minimum single-leg width |
| LEG_THICK | 0.0567m² | Maximum single-leg width |
| TWO_LEGS_THIN | 0.056644m² | Minimum two-leg cluster width |
| TWO_LEGS_THICK | 0.25m² | Maximum two-leg cluster width |

### Stage 4: Geometric Validation

```cpp
is_leg()
```

- Calculates angle between line-of-sight and leg tangent
- Requires angle > 0.5 radians for leg classification
- Centers must be within 3m from robot

### Stage 5: In-Front Zone Detection

```cpp
get_nearest_legs_in_front()
```

**Valid Region (relative to robot):**
- X: 0.25m to 1.5m (forward)
- Y: -0.5m to +0.5m (lateral)
- Selects closest detection by Euclidean distance

### Stage 6: Temporal Tracking & Filtering

**Butterworth IIR Filter (4th order):**
- X-axis cutoff: 0.7 Hz
- Y-axis cutoff: 0.2 Hz
- First detection confirmed after 20 consecutive frames
- Lost tracking after 20 frames without detection

## Parameters

**launch.xml Parameters:**

```xml
<param name="scan_downsampling" value="1"/>
<param name="show_hypothesis" value="false"/>
<param name="laser_scan_frame" value="base_range_sensor_link"/>
<param name="laser_scan_topic" value="/scan"/>
<param name="base_link_frame" value="base_footprint"/>
```

## Topics

### Published
| Topic | Type | Description |
|-------|------|-------------|
| `/hri/leg_finder/leg_pose` | PointStamped | Filtered leg position |
| `/hri/leg_finder/legs_found` | Bool | Leg tracking status |
| `/hri/leg_finder/hypothesis` | Marker | Debug visualization |

### Subscribed
| Topic | Type | Description |
|-------|------|-------------|
| `/scan` | LaserScan | LIDAR scan input |
| `/hri/leg_finder/enable` | Bool | Enable detection |
| `/stop` | Empty | Emergency stop |

## TF2 Frames

- Source: `laser_scan_frame` (e.g., "base_range_sensor_link")
- Target: `base_link_frame` (e.g., "base_footprint")
- Transform wait: 10 seconds

## Tuning

| Constant | Value | Effect |
|----------|-------|--------|
| FILTER_THRESHOLD | 0.081 | Aggressiveness of range smoothing |
| FLANK_THRESHOLD | 0.04 | Sensitivity to leg edges |
| IS_LEG_THRESHOLD | 0.5 | Min angle for geometric validation |
| HORIZON_THRESHOLD | 9 | Max range (m²) |
