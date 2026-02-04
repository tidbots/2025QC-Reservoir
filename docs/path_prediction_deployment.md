# デプロイメントガイド

Docker環境でのpath_predictionシステムのセットアップと実行。

## システム要件

- Ubuntu 22.04
- Docker + Docker Compose
- NVIDIA GPU（オプション、推奨）
- NVIDIA Container Toolkit（GPU使用時）

## ファイル構成

```
path_prediction/
├── Dockerfile                    # コンテナ定義
├── compose.yaml                  # GPU対応compose
├── compose.no-gpu.yaml           # CPU専用compose
├── docker_build.sh               # ビルドヘルパー
└── assets/
    ├── cyclonedds_profile.xml    # DDSネットワーク設定
    ├── terminator_config         # ターミナル設定
    └── entrypoint.sh             # コンテナ起動スクリプト
```

## 初回セットアップ

### 1. ビルドスクリプトの実行

```bash
cd path_prediction
./docker_build.sh --user=<username> --robot=<robot_name> --netif=<network_interface>
```

**パラメータ:**
| パラメータ | デフォルト | 説明 |
|-----------|---------|------|
| `--user` | roboworks | ユーザー名 |
| `--uid` | 1000 | ユーザーID |
| `--gid` | 1000 | グループID |
| `--password` | tamagawa | パスワード |
| `--robot` | hsrc30 | ロボット識別子 |
| `--netif` | enp3s0 | ネットワークインターフェース |
| `--rosip` | 192.168.103.4 | ROS IPアドレス |
| `--domain` | 30 | ROS Domain ID |

### 2. Dockerイメージのビルド

```bash
docker compose build
```

### 3. コンテナの起動

```bash
# GPU有効
docker compose up

# CPU専用
docker compose -f compose.no-gpu.yaml up
```

## CycloneDDS設定

**ファイル:** `assets/cyclonedds_profile.xml`

```xml
<Discovery>
  <Peers>
    <Peer Address="192.168.103.30"/>  <!-- ロボットIP: 要変更 -->
    <Peer Address="localhost"/>
  </Peers>
</Discovery>
```

**重要:** `<Peer Address>`を実際のロボット/PCのIPアドレスに更新してください。

### 設定手順

1. ネットワーク上のロボットIPを特定
2. XMLファイルの`<Peer Address>`を更新
3. `docker_build.sh`で`--rosip`をコンテナIPに設定
4. Dockerイメージを再ビルド

## Docker Compose設定

### GPU設定（compose.yaml）

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          capabilities: [gpu]
```

### ネットワーク設定

```yaml
network_mode: "host"    # DDSに必須
ipc: host               # 共有メモリ
privileged: true        # デバイスアクセス
```

### X11ディスプレイ転送

```yaml
environment:
  - DISPLAY=${DISPLAY}
volumes:
  - /tmp/.X11-unix:/tmp/.X11-unix:rw
  - ~/.Xauthority:/home/${USER_NAME}/.Xauthority:ro
```

## 環境変数

コンテナ内で設定される主要な環境変数:

```bash
ROS_DOMAIN_ID=30
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
CYCLONEDDS_URI=/home/$USER_NAME/.config/cyclonedds_profile.xml
ROS_IP=<設定値>
LANG=ja_JP.UTF-8
TZ=Asia/Tokyo
```

## 再ビルド

```bash
# .envファイルを使用して再ビルド
docker compose build

# キャッシュなしで完全再ビルド
docker compose build --no-cache
```

## トラブルシューティング

### 通信問題

**症状:** トピックが見えない、ノードが通信していない

**解決策:**
1. `ROS_DOMAIN_ID`を確認: `echo $ROS_DOMAIN_ID`
2. CycloneDDS設定のピアIPを確認
3. ネットワークインターフェースを確認: `ip a`
4. 接続テスト: `ping <robot_ip>`

### GPU問題

**症状:** GPU認識されない

**解決策:**
1. NVIDIA Container Toolkitをインストール
2. `nvidia-smi`で動作確認
3. `compose.yaml`を使用（`compose.no-gpu.yaml`ではなく）

### 表示問題

**症状:** GUIアプリが表示されない

**解決策:**
```bash
xhost +local:docker
```
