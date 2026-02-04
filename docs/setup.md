# セットアップガイド

## Dockerセットアップ（初回）

```bash
cd path_prediction
./docker_build.sh --user=roboworks --robot=hsrc30 --netif=<network-interface>
docker compose up           # GPU有効
docker compose -f compose.no-gpu.yaml up  # CPUのみ
```

## コンテナ内での操作

```bash
# ROSワークスペースのビルド
cd ~/ros2_ws && colcon build --symlink-install
source install/setup.bash

# 脚検出器の起動
ros2 launch leg_finder leg_finder.launch.xml

# 検出の有効化
ros2 topic pub --once /hri/leg_finder/enable std_msgs/msg/Bool "data: true"

# 経路予測の実行
ros2 run esn_path_prediction esn_path_prediction.py
```

## Dockerイメージの再ビルド

```bash
docker compose build
```
