# leg_finder ノード詳細

C++によるリアルタイム2D LIDARベースの脚検出ノード。

**ファイル:** `path_prediction/ros2_ws/src/leg_finder/src/leg_finder_node.cpp`

## 検出アルゴリズム

### ステージ1: レーザーレンジフィルタリング

```cpp
filter_laser_ranges()
```

- FILTER_THRESHOLD = 0.081m でノイズ除去
- 2点または3点の移動平均を適用
- 孤立点やノイズをゼロ化

### ステージ2: ダウンサンプリング（オプション）

```cpp
downsample_scan()
```

- `scan_downsampling`パラメータで制御（デフォルト: 1）
- 計算負荷軽減のためスキャンポイントをスキップ

### ステージ3: 脚仮説検出

```cpp
find_leg_hypothesis()
```

1. 極座標(range, angle) → 直交座標(x, y)に変換
2. TF2でレーザーフレームからbase_linkフレームに変換
3. フランク検出（急激な距離変化 > 0.04m）
4. 幾何学的基準で脚候補を検証

**脚サイズ制約:**
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| LEG_THIN | 0.00341m² | 単脚の最小幅 |
| LEG_THICK | 0.0567m² | 単脚の最大幅 |
| TWO_LEGS_THIN | 0.056644m² | 二脚クラスタの最小幅 |
| TWO_LEGS_THICK | 0.25m² | 二脚クラスタの最大幅 |

### ステージ4: 幾何学的検証

```cpp
is_leg()
```

- 視線と脚接線の角度を計算
- 角度 > 0.5 radians で脚と判定
- 中心がロボットから3m以内

### ステージ5: 前方領域検出

```cpp
get_nearest_legs_in_front()
```

**有効領域（ロボット基準）:**
- X: 0.25m ～ 1.5m（前方）
- Y: -0.5m ～ +0.5m（横方向）
- ユークリッド距離で最も近い検出を選択

### ステージ6: 時間的追跡とフィルタリング

**バターワースIIRフィルタ（4次）:**
- X軸カットオフ: 0.7 Hz
- Y軸カットオフ: 0.2 Hz
- 20フレーム連続で初回検出確定
- 20フレーム検出なしで追跡ロスト

## パラメータ

**launch.xmlパラメータ:**

```xml
<param name="scan_downsampling" value="1"/>
<param name="show_hypothesis" value="false"/>
<param name="laser_scan_frame" value="base_range_sensor_link"/>
<param name="laser_scan_topic" value="/scan"/>
<param name="base_link_frame" value="base_footprint"/>
```

## トピック

### Publish
| トピック | 型 | 説明 |
|----------|-----|------|
| `/hri/leg_finder/leg_pose` | PointStamped | フィルタ済み脚位置 |
| `/hri/leg_finder/legs_found` | Bool | 脚追跡状態 |
| `/hri/leg_finder/hypothesis` | Marker | デバッグ可視化 |

### Subscribe
| トピック | 型 | 説明 |
|----------|-----|------|
| `/scan` | LaserScan | LIDARスキャン入力 |
| `/hri/leg_finder/enable` | Bool | 検出有効化 |
| `/stop` | Empty | 緊急停止 |

## TF2フレーム

- ソース: `laser_scan_frame`（例: "base_range_sensor_link"）
- ターゲット: `base_link_frame`（例: "base_footprint"）
- 変換待機: 10秒

## チューニング

| 定数 | 値 | 効果 |
|------|-----|------|
| FILTER_THRESHOLD | 0.081 | レンジ平滑化の積極性 |
| FLANK_THRESHOLD | 0.04 | 脚エッジの感度 |
| IS_LEG_THRESHOLD | 0.5 | 幾何学検証の最小角度 |
| HORIZON_THRESHOLD | 9 | 最大距離（m²） |
