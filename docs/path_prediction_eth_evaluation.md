# ETHデータセットによるESN評価

ETH歩行者追跡データセットを使用したESN経路予測の検証。

## 概要

ETHデータセットは、チューリッヒ工科大学が公開している歩行者追跡のベンチマークデータセットです。実際の歩行者の軌跡データを使用することで、シミュレーションパターンよりも現実的な評価が可能になります。

## データセット

### students001_train.txt
- 場所: ETHキャンパス
- 歩行者数: 400+
- フォーマット: `frame  ped_id  x  y`
- 座標系: メートル単位

### biwi_eth.txt
- BIWI (歩行行動研究所) データセット
- 同様のフォーマット

## 評価ツール

### バッチ評価 (eth_esn_batch.py)

GUIなしでESN予測精度を評価します。

```bash
# デフォルト評価（5人の歩行者）
python3 tools/eth_esn_batch.py

# 特定の歩行者IDを指定
python3 tools/eth_esn_batch.py --ped_ids 399 168 269 177 178

# パラメータ調整
python3 tools/eth_esn_batch.py --n_models 10 --future_horizon 20
```

**パラメータ:**
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| --data | data/students001_train.txt | データセットパス |
| --ped_ids | 自動選択 | 評価する歩行者ID |
| --n_peds | 5 | 自動選択時の歩行者数 |
| --n_models | 10 | ESNモデル数 |
| --future_horizon | 20 | 予測ステップ数 |

### 可視化 (eth_esn_visualizer.py)

予測結果を可視化します。

```bash
# 可視化実行
python3 tools/eth_esn_visualizer.py --ped_ids 399 168 269

# 出力先指定
python3 tools/eth_esn_visualizer.py --output output
```

**出力:**
- 各歩行者の軌跡と予測結果
- 複数フレームでの予測比較
- サマリー統計

## 評価結果

### テスト条件
- ESNモデル数: 10
- 予測ホライズン: 20ステップ
- ウォームアップ: 5フレーム
- ウィンドウサイズ: 20フレーム

### 結果サマリー

| 歩行者ID | 平均誤差 (m) | 標準偏差 | フレーム数 |
|---------|-------------|---------|-----------|
| 399 | 0.620 | 0.346 | 301 |
| 168 | 1.094 | 1.146 | 202 |
| 269 | 1.014 | 0.508 | 191 |
| 177 | 0.845 | 0.857 | 184 |
| 178 | 0.931 | 0.543 | 184 |
| **平均** | **0.901** | | |

### 可視化例

#### 歩行者399の予測結果
![ETH Pedestrian 399](images/eth_ped_399.png)

#### 評価サマリー
![ETH Summary](images/eth_summary.png)

## V1 vs V2 比較

### 比較ツール (eth_v1_v2_comparison.py)

V1（オリジナルESN）とV2（カルマンハイブリッド）をETHデータセットで比較。

```bash
python3 tools/eth_v1_v2_comparison.py --ped_ids 399 168 269 177 178
```

### 比較結果

| 歩行者ID | V1 (ESN) | V2 (Kalman Hybrid) | 改善率 |
|---------|----------|-------------------|-------|
| 399 | 0.620m | 0.578m | +6.8% |
| 168 | 1.094m | 0.888m | +18.8% |
| 269 | 1.014m | 0.851m | +16.1% |
| 177 | 0.845m | 0.692m | +18.1% |
| 178 | 0.931m | 0.755m | +18.9% |
| **平均** | **0.901m** | **0.753m** | **+16.4%** |

### 比較可視化

![V1 vs V2 Comparison](images/eth_v1_v2_comparison.png)

### 結論

- V2（カルマンハイブリッド）は実データでも**平均16.4%の改善**
- シミュレーション（+18.8%）とほぼ同等の効果
- 全歩行者で改善を確認

---

## 考察

### シミュレーションとの比較

| 評価方法 | 平均誤差 | 予測ホライズン |
|---------|---------|---------------|
| シミュレーション（合成パターン） | 0.14-0.27m | 20ステップ |
| ETHデータセット | 0.62-1.09m | 20ステップ |

ETHデータセットでの誤差が大きい理由:
1. **座標スケール**: ETHは実世界のメートル単位（〜15m範囲）
2. **軌跡の複雑さ**: 実際の歩行者は予測困難な動きをする
3. **速度変動**: 停止、加速、方向転換が頻繁

### ESNの適用性

- 長い軌跡（300+フレーム）では誤差が小さい傾向
- オンライン学習により軌跡パターンに適応
- 20ステップ先予測で平均1m未満の誤差は実用的

## ファイル構成

```
tools/
├── data/
│   ├── students001_train.txt  # ETHデータセット
│   └── biwi_eth.txt           # BIWIデータセット
├── eth_esn_batch.py           # バッチ評価スクリプト
├── eth_esn_visualizer.py      # 可視化スクリプト
└── person_tracking_esn_fx.py  # オリジナルスクリプト
```

## 参考文献

- ETH Walking Pedestrians Dataset: https://icu.ee.ethz.ch/research/datsets.html
- Pellegrini, S., et al. "You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking." ICCV 2009.
