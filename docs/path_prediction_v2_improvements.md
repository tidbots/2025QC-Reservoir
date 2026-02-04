# ESN経路予測 V2 改良検証

path_prediction_v2での予測精度改善に関する検証記録。

## 改良アプローチ

### カルマンフィルタハイブリッド

ESNの予測結果とカルマンフィルタの予測を重み付けで組み合わせる手法。

**実装:**
```python
class SimpleKalmanFilter:
    """2D位置追跡用カルマンフィルタ - 状態: [x, y, vx, vy]"""
    def __init__(self, dt=0.1, process_noise=0.1, measurement_noise=0.05):
        # 状態遷移行列、観測行列、共分散行列

class KalmanHybridESNPredictor:
    """ESN + カルマンフィルタのハイブリッド"""
    # 重み付け: (1 - kalman_weight) * esn_pred + kalman_weight * kalman_pred
    # kalman_weight = 0.3
```

**結果:** ETHデータセットで **+16.4%** 改善

## ETHデータセットでの検証結果

### V1 vs V2 比較

| 歩行者ID | V1 (ESN) | V2 (Kalman Hybrid) | 改善率 |
|---------|----------|-------------------|-------|
| 399 | 0.620m | 0.578m | +6.8% |
| 168 | 1.094m | 0.888m | +18.8% |
| 269 | 1.014m | 0.851m | +16.1% |
| 177 | 0.845m | 0.692m | +18.1% |
| 178 | 0.931m | 0.755m | +18.9% |
| **平均** | **0.901m** | **0.753m** | **+16.4%** |

### V1 vs V2 比較グラフ

![V1 vs V2 Comparison](images/eth_v1_v2_comparison.png)

### 評価サマリー

![ETH Summary](images/eth_summary.png)

### 予測例（歩行者399）

![ETH Pedestrian 399](images/eth_ped_399.png)

## 試行した他のアプローチ

以下のアプローチは効果が限定的または悪化したため不採用：

### 1. リザバーサイズの拡大
- V1: 25ユニット → V2: 50ユニット
- 結果: 精度悪化（学習データ不足）

### 2. スペクトル半径の増加
- V1: 0.8-0.9 → V2: 0.88-0.95
- 結果: 不安定化（オンライン学習との相性が悪い）

### 3. 方向転換検出のチューニング
- 角度・速度しきい値による検出
- 結果: 一部パターンでのみ有効

## 結論

- **カルマンフィルタハイブリッド**が最も効果的な改良
- ESNの短期予測力とカルマンフィルタの平滑化効果の組み合わせが有効
- 実データ（ETHデータセット）で16.4%の改善を確認

## ファイル構成

```
tools/
├── eth_esn_batch.py           # バッチ評価スクリプト
├── eth_esn_visualizer.py      # 可視化スクリプト
├── eth_v1_v2_comparison.py    # V1 vs V2比較スクリプト
└── data/
    ├── students001_train.txt  # ETHデータセット
    └── biwi_eth.txt           # BIWIデータセット
```

## 再現方法

```bash
# V1 vs V2比較
python3 tools/eth_v1_v2_comparison.py --ped_ids 399 168 269 177 178

# バッチ評価
python3 tools/eth_esn_batch.py --ped_ids 399 168 269

# 可視化
python3 tools/eth_esn_visualizer.py --ped_ids 399 168 269
```
