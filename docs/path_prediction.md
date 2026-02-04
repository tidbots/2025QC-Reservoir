# Path Prediction システム概要

ROS 2ベースの人間-ロボットインタラクション（HRI）システム。LIDARを用いた人間の脚検出とEcho State Networks（ESN）による経路予測を行う。

## システム構成

```
LIDARスキャン (/scan)
    ↓
leg_finder (C++) - 幾何学的検出 + バターワースフィルタ
    ↓ /hri/leg_finder/leg_pose (PointStamped)
esn_path_prediction (Python) - ESNアンサンブル + オンライン学習
    ↓ /hri/leg_finder/predicted_path (nav_msgs/Path)
```

## 主要パッケージ

| パッケージ | 言語 | 場所 | 目的 |
|---------|----------|----------|---------|
| leg_finder | C++ | `path_prediction/ros2_ws/src/leg_finder/` | レーザースキャンからの脚検出 |
| esn_path_prediction | Python | `path_prediction/ros2_ws/src/esn_path_prediction/` | ESNによる経路予測 |

## ROSトピック

### 入力
- `/scan` (sensor_msgs/LaserScan) - LIDARスキャンデータ

### 制御
- `/hri/leg_finder/enable` (std_msgs/Bool) - 検出の有効化/無効化
- `/stop` (std_msgs/Empty) - 緊急停止

### 出力
- `/hri/leg_finder/leg_pose` (geometry_msgs/PointStamped) - 検出した脚の位置
- `/hri/leg_finder/legs_found` (std_msgs/Bool) - 脚検出状態
- `/hri/leg_finder/predicted_path` (nav_msgs/Path) - 予測経路（20ステップ先）
- `/hri/leg_finder/hypothesis` (visualization_msgs/Marker) - デバッグ用可視化

## 技術スタック

- Ubuntu 22.04 + CUDA 11.7.1
- ROS 2 Humble + CycloneDDS
- ReservoirPy（Echo State Networks）
- Docker + NVIDIA GPU対応

## クイックスタート

```bash
# Dockerセットアップ
cd path_prediction
./docker_build.sh --user=roboworks --robot=hsrc30 --netif=<network-interface>
docker compose up

# コンテナ内で実行
cd ~/ros2_ws && colcon build --symlink-install && source install/setup.bash

# ターミナル1: 脚検出器
ros2 launch leg_finder leg_finder.launch.xml

# ターミナル2: 検出有効化
ros2 topic pub --once /hri/leg_finder/enable std_msgs/msg/Bool "data: true"

# ターミナル3: 経路予測
ros2 run esn_path_prediction esn_path_prediction.py
```

## 詳細ドキュメント

- [leg_finder詳細](path_prediction_leg_finder.md) - 脚検出アルゴリズム
- [ESN経路予測詳細](path_prediction_esn.md) - ESNアルゴリズム
- [デプロイメント](path_prediction_deployment.md) - Docker設定
