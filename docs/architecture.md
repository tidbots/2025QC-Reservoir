# アーキテクチャ

## システム構成

```
LaserScan (/scan)
    ↓
leg_finder (C++) - 幾何学的検出 + バターワースフィルタリング
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

- **入力**: `/scan` (LaserScan)
- **制御**: `/hri/leg_finder/enable` (Bool), `/stop` (Empty)
- **出力**: `/hri/leg_finder/leg_pose` (PointStamped), `/hri/leg_finder/predicted_path` (Path)

## 技術スタック

- Ubuntu 22.04 + CUDA 11.7.1
- ROS 2 Humble with CycloneDDS
- ReservoirPy (Echo State Networks)
- Docker with NVIDIA GPU support

## 主要設定

- ROS Domain ID: 30 (`--domain`で設定可能)
- CycloneDDS: マルチキャスト無効、ピアベース探索
- ESN: 10モデルのアンサンブル、20ステップ予測ホライズン
