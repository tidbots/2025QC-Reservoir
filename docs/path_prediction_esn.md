# ESN経路予測ノード詳細

PythonによるEcho State Networks（ESN）を用いた経路予測ノード。

**ファイル:** `path_prediction/ros2_ws/src/esn_path_prediction/esn_path_prediction/esn_path_prediction.py`

## アーキテクチャ

**フレームワーク:** ReservoirPy + scikit-learn + scipy

**アンサンブルモデル:** 複数のESN（デフォルト: 10）の予測を平均化。

**パイプライン:**
1. 入力平滑化（Savitzky-Golay）
2. オンライン標準化（EWMA）
3. マルチモデルESN推論
4. オンライン適応/再学習
5. クリッピングと安定化

## 主要コンポーネント

### OnlineStandardizer クラス

指数加重移動平均（EWMA）によるオンラインZ-score正規化。

```python
class OnlineStandardizer:
    def __init__(self, mean, var, alpha=0.02):
        self.alpha = 0.02  # 学習率
```

### 多様なESN生成

```python
create_diverse_esns(n_models=5, base_units=25, seed=42, rls_forgetting=0.99)
```

**ESNごとのランダム化:**
- リザバーユニット: base_units ± random(-5, 5)
- スペクトル半径: 0.8-0.9
- リーク率: 0.35-0.6
- 入力スケーリング: 0.2-0.4
- バイアス: -0.2 ～ +0.2

### Savitzky-Golay平滑化

```python
savgol_win(win, window_length=9, polyorder=2)
```

ESN入力前の脚位置履歴のノイズ除去。

## ROSパラメータ

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|---------|------|
| `legs_topic` | string | `/hri/leg_finder/leg_pose` | 入力脚位置トピック |
| `frame_id` | string | `base_footprint` | 出力フレーム |
| `warmup` | int | 5 | ESN初期化に必要なサンプル数 |
| `window` | int | 20 | 履歴バッファの最大サイズ |
| `future_horizon` | int | 20 | 予測ステップ数 |
| `n_models` | int | 10 | アンサンブル内のESN数 |
| `leg_update_hz` | float | 10.0 | 脚更新の最大周波数 |
| `update_rate_hz` | float | 20.0 | 予測発行レート |

## 内部チューニングパラメータ

| パラメータ | 値 | 目的 |
|-----------|-----|------|
| `sg_window` | 9 | Savitzky-Golayウィンドウ |
| `sg_poly` | 2 | SGの多項式次数 |
| `adapt_window` | 5 | 適応用の直近データウィンドウ |
| `sudden_change_thresh` | 0.6 | ESN状態リセットの距離閾値 |
| `adapt_damping_nominal` | 0.35 | 通常時のRLS学習率 |
| `adapt_damping_boost` | 1.0 | エラー高時のRLS学習率 |
| `boost_error_thresh` | 0.5 | ブーストモード発動の閾値 |
| `state_clip` | 5.0 | リザバー状態のクリップ限界 |
| `wout_clip` | 8.0 | 出力重みのクリップ限界 |

## ワークフロー

### フェーズ1: ウォームアップ（0-5サンプル）

```python
if len(self.history) >= max(self.warmup, 6):
    # 履歴でStandardScalerをフィット
    # OnlineStandardizersを初期化
    # ESNをウォームスタート
```

### フェーズ2: 脚受信コールバック

1. 周波数ゲーティング（≤10Hz）
2. 位置抽出
3. 履歴・バッファ更新
4. Savitzky-Golay平滑化
5. オンライン標準化器の更新

### フェーズ3: 予測ループ（20Hz）

1. 急激な変化の検出（閾値 > 0.6でリセット）
2. マルチステップロールアウト（各ESNで20ステップ）
3. オンライン適応（エラーベースのブーストモード）
4. クリッピングと安定化
5. アンサンブル平均

## トピック

### Publish
| トピック | 型 | 説明 |
|----------|-----|------|
| `/hri/leg_finder/predicted_path` | nav_msgs/Path | 20ステップの将来軌道 |

### Subscribe
| トピック | 型 | 説明 |
|----------|-----|------|
| `/hri/leg_finder/leg_pose` | PointStamped | leg_finderからの検出位置 |

## チューニングガイド

**高速収束向け:**
- `n_models`: 10 → 5に削減
- `leg_update_hz`: 10 → 20に増加
- `adapt_damping_boost`: 1.0 → 2.0に増加

**安定性向上（ノイズ環境）:**
- `sg_window`: 9 → 11または13に増加
- `sg_poly`: 2 → 1に減少
- `adapt_damping_nominal`: 0.35 → 0.15に減少

**長距離予測向け:**
- `future_horizon`: 20 → 30または40に増加
- `window`: 20 → 30に増加
