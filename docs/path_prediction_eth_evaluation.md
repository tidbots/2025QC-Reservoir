# ETHデータセットによるESN評価

ETH歩行者追跡データセットを使用したESN経路予測の検証。

## 概要

ETHデータセットは、チューリッヒ工科大学が公開している歩行者追跡のベンチマークデータセットです。実際の歩行者の軌跡データを使用してESN経路予測の精度を評価します。

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

- V2（カルマンハイブリッド）は**平均16.4%の改善**
- 全歩行者で改善を確認
- カルマンフィルタの平滑化効果が有効

---

## V3 適応型ESN

### 概要

V3は軌跡の複雑さに応じてESNとカルマンフィルタの重みを動的に調整します。

**主な特徴:**
- **軌跡複雑度分析**: 方向変化と速度変動からスコアを計算
- **適応的重み調整**: 複雑な軌跡ではESNの重みを増加
- **性能ベースフィードバック**: 最近の予測誤差に基づく重み調整

### 比較ツール (eth_v3_adaptive.py)

```bash
python3 tools/eth_v3_adaptive.py --ped_ids 399 168 269 177 178
```

### 比較結果

| 歩行者ID | V1 (ESN) | V2 (Kalman Hybrid) | V3 (Adaptive) | vs V1 | vs V2 |
|---------|----------|-------------------|---------------|-------|-------|
| 399 | 0.620m | 0.578m | 0.453m | +26.9% | +21.6% |
| 168 | 1.094m | 0.888m | 0.742m | +32.2% | +16.4% |
| 269 | 1.014m | 0.851m | 0.698m | +31.2% | +18.0% |
| 177 | 0.845m | 0.692m | 0.589m | +30.3% | +14.9% |
| 178 | 0.931m | 0.755m | 0.635m | +31.8% | +15.9% |
| **平均** | **0.901m** | **0.753m** | **0.623m** | **+30.8%** | **+17.2%** |

### 可視化

![V3 Adaptive Comparison](images/eth_v3_adaptive.png)

### 考察

1. **V3はV1比30.8%、V2比17.2%の改善**
2. **ESNの適応学習が効果を発揮**
   - 複雑な軌跡で高いESN重みを使用
   - 直線的な軌跡ではカルマンフィルタを優先
3. **軌跡複雑度に応じた動的な重み調整が有効**

---

## 従来手法との比較

### 比較ツール (eth_method_comparison.py)

ESNと従来の軌道予測手法を比較。

```bash
python3 tools/eth_method_comparison.py --ped_ids 399 168 269 177 178
```

### 比較手法

| 手法 | 説明 | 出典 |
|-----|------|------|
| Linear | 線形外挿 | - |
| f(x) avg | 線形+放物線+シグモイドの平均 | RSJ2025 1I5-03 |
| Kalman | カルマンフィルタのみ | - |
| ESN | Echo State Network アンサンブル | - |
| ESN+Kalman | ESN + カルマンハイブリッド | 本プロジェクト |

### 比較結果

| 手法 | 平均誤差 (m) | vs Linear |
|-----|-------------|----------|
| **Kalman** | **0.509** | **+25.6%** |
| Linear | 0.684 | - |
| ESN+Kalman | 0.753 | -10.1% |
| ESN | 0.901 | -31.7% |
| f(x) avg | 3.443 | -403.5% |

### 比較可視化

![Method Comparison](images/eth_method_comparison.png)

### 考察

1. **カルマンフィルタ単体**が最も良い結果
   - 速度ベースの予測が歩行者の直線的な動きに有効
   - 計算コストも低い

2. **f(x)平均**は不安定
   - シグモイドフィッティングが発散するケースあり
   - 論文の手法は特定条件下では有効だが汎用性に課題

3. **ESN+Kalman**はESN単体より改善
   - カルマンフィルタの安定性がESNの予測を補完
   - 複雑な軌跡ではESNの適応学習が有効な可能性

4. **ESN単体**は従来手法より劣る場合あり
   - オンライン学習の収束に時間が必要
   - 単純な直線軌道では過学習の可能性

---

## 考察

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
├── eth_v1_v2_comparison.py    # V1 vs V2比較
├── eth_v3_adaptive.py         # V3適応型ESN評価
├── eth_method_comparison.py   # 従来手法との比較
└── person_tracking_esn_fx.py  # オリジナルスクリプト
```

## 参考文献

- ETH Walking Pedestrians Dataset: https://icu.ee.ethz.ch/research/datsets.html
- Pellegrini, S., et al. "You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking." ICCV 2009.
- 小野, 崔. "四輪独立駆動型全方向移動ロボットを用いたMPPI制御による歩行者回避." RSJ2025, 1I5-03. (f(x)予測手法の参考)
