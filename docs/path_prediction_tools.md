# 検証ツール

ETH歩行者追跡データセットを使用したESN経路予測の検証・可視化ツール。

## ETHデータセット

### データファイル

| ファイル | 説明 |
|---------|------|
| `tools/data/students001_train.txt` | ETHキャンパスの歩行者データ |
| `tools/data/biwi_eth.txt` | BIWIデータセット |

### データフォーマット

```
frame  ped_id  x  y
```
- frame: フレーム番号
- ped_id: 歩行者ID
- x, y: 位置座標（メートル）

---

## バッチ評価 (eth_esn_batch.py)

GUIなしでESN予測精度を評価。

### 使用方法

```bash
# デフォルト評価（5人の歩行者）
python3 tools/eth_esn_batch.py

# 特定の歩行者IDを指定
python3 tools/eth_esn_batch.py --ped_ids 399 168 269 177 178

# パラメータ調整
python3 tools/eth_esn_batch.py --n_models 10 --future_horizon 20
```

### パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| --data | data/students001_train.txt | データセットパス |
| --ped_ids | 自動選択 | 評価する歩行者ID |
| --n_peds | 5 | 自動選択時の歩行者数 |
| --n_models | 10 | ESNモデル数 |
| --future_horizon | 20 | 予測ステップ数 |

---

## 可視化 (eth_esn_visualizer.py)

予測結果を可視化。

### 使用方法

```bash
python3 tools/eth_esn_visualizer.py --ped_ids 399 168 269
```

### 出力

- 各歩行者の軌跡と予測結果
- 複数フレームでの予測比較
- 統計サマリー

---

## V1 vs V2 比較 (eth_v1_v2_comparison.py)

V1（オリジナルESN）とV2（カルマンハイブリッド）を比較。

### 使用方法

```bash
python3 tools/eth_v1_v2_comparison.py --ped_ids 399 168 269 177 178
```

### 出力

- 歩行者ごとの誤差比較
- 改善率の可視化
- 全体サマリー

---

## オリジナルスクリプト (person_tracking_esn_fx.py)

同僚が作成したESN評価スクリプト。アニメーション機能付き。

### 使用方法

```bash
# MP4アニメーション保存
python3 tools/person_tracking_esn_fx.py --save_mp4 --ped_ids 399
```

---

## PDF変換 (md2pdf.py)

MarkdownドキュメントをPDFに変換。

### 使用方法

```bash
python3 tools/md2pdf.py docs/path_prediction_eth_evaluation.md
```

---

## ファイル構成

```
tools/
├── data/
│   ├── students001_train.txt    # ETHデータセット
│   └── biwi_eth.txt             # BIWIデータセット
├── eth_esn_batch.py             # バッチ評価
├── eth_esn_visualizer.py        # 可視化
├── eth_v1_v2_comparison.py      # V1 vs V2比較
├── person_tracking_esn_fx.py    # オリジナルスクリプト
└── md2pdf.py                    # PDF変換
```
