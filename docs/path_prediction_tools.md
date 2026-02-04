# 可視化ツール

## ESN Visualizer

ROSなしでESN経路予測アルゴリズムをテスト・可視化するスタンドアロンスクリプト。

**ファイル:** `tools/esn_visualizer.py`

### 使用方法

```bash
# 全パターンをテスト
python3 tools/esn_visualizer.py --pattern all --output output

# 特定パターンのみ
python3 tools/esn_visualizer.py --pattern straight --output output
```

### テストパターン

| パターン | 説明 | 特徴 |
|---------|------|------|
| `straight` | 直線歩行 | 微小な横揺れあり |
| `curve` | 曲線歩行 | 円弧軌道 |
| `zigzag` | ジグザグ歩行 | 周期的な横方向移動 |
| `stop_and_go` | 停止と移動 | 停止期間を含む |

### 出力

各パターンに対して4つのグラフを含むPNG画像を生成：

1. **軌跡と予測の全体像** - 実際の軌跡（青）と予測（赤系）
2. **予測詳細** - 特定時点での過去・現在・将来・予測の比較
3. **予測誤差の推移** - 時系列での誤差変化
4. **X/Y成分別精度** - 成分ごとの予測精度

### 実装

スクリプトには以下のスタンドアロン実装が含まれています：

- `SimpleESN` - Echo State Networkの簡易実装
- `OnlineStandardizer` - オンラインZ-score正規化
- `ESNPredictor` - ESN予測器（ROS非依存）

これらは`esn_path_prediction.py`のアルゴリズムを再現しています。

### 結果例

**予測精度（平均誤差）:**

| パターン | 平均誤差 | 標準偏差 |
|---------|---------|---------|
| straight | ~0.13 m | ~0.09 m |
| curve | ~0.21 m | ~0.18 m |
| zigzag | ~0.36 m | ~0.25 m |
| stop_and_go | ~0.18 m | ~0.18 m |

※ランダムシードにより結果は変動します

### カスタマイズ

`ESNPredictor`クラスのパラメータを変更可能：

```python
predictor = ESNPredictor(
    n_models=10,        # ESNの数
    warmup=5,           # ウォームアップサンプル数
    window=20,          # 履歴ウィンドウサイズ
    future_horizon=20   # 予測ステップ数
)
```

---

## ETH Dataset Tools

ETH歩行者追跡データセットを使用した評価・可視化ツール。

詳細は [ETHデータセット評価](path_prediction_eth_evaluation.md) を参照してください。

### バッチ評価 (eth_esn_batch.py)

GUIなしでESN予測精度を評価。

```bash
# デフォルト評価
python3 tools/eth_esn_batch.py

# 歩行者ID指定
python3 tools/eth_esn_batch.py --ped_ids 399 168 269
```

### 可視化 (eth_esn_visualizer.py)

ETHデータセットでの予測結果を可視化。

```bash
python3 tools/eth_esn_visualizer.py --ped_ids 399 168 269
```

**出力:**
- 歩行者ごとの軌跡と予測結果
- 複数フレームでの予測比較
- 統計サマリー

---

## 改良検証ツール (esn_improvement_test.py)

漸進的な改良アプローチを検証。

```bash
python3 tools/esn_improvement_test.py --output output
```

**検証内容:**
- 方向転換検出のチューニング
- カルマンフィルタハイブリッド
- 複合アプローチ

詳細は [V2改良検証](path_prediction_v2_improvements.md) を参照。
