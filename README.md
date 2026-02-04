# 2025QC-Reservoir
リザバーコンピューティングの有効性評価のための検証支援業務

## 概要
国立研究開発法人新エネルギー・産業技術総合開発機構（NEDO）の「リザバーコンピューティング技術の有効性検証に関する技術調査事業」において、現場データを取得し有効性を評価するためのデータ作成と弊社が提供するライブラリを活用した分析の試行的実施を行う。

## 業務内容
リザバーコンピューティングの家庭用サービスロボット分野での活用の方法を検討し、検証用データの作成と弊社が提供するライブラリを活用した技術の有効性の評価を行う。

具体的には、以下の作業を行う。
- リザバーを用いた移動人物の経路予測プログラムの作成
- 家庭環境での従来手法との比較調査
- ＱｕａｎｔｕｍＣｏｒｅ社開発のライブラリの統合検討

## 成果物
- 検証結果報告書
- 検証用データ
- 検証用プログラム及びマニュアル　一式

## TID -> QC
[ROS 2 Leg Finder + Path Prediction (Dockerized)](https://gitlab.com/tidbots/path_prediction)

```
git clone --recursive git@github.com:tidbots/2025QC-Reservoir.git
```

## 評価経過

### ETHデータセットによる予測精度検証

ETH歩行者追跡データセット（ETH Zurich公開）を使用してESN経路予測の精度を評価。

#### 手法比較結果

| 手法 | 平均誤差 (m) | 備考 |
|------|-------------|------|
| **Kalman単体** | **0.509** | 直線軌道に最適 |
| Linear | 0.684 | 線形外挿 |
| V3 (Adaptive ESN) | 0.623 | 軌跡複雑度に応じた動的重み調整 |
| V2 (ESN+Kalman) | 0.753 | 固定重みハイブリッド |
| V1 (ESN only) | 0.901 | ESNアンサンブル |
| f(x) avg | 3.443 | RSJ2025 1I5-03の手法（不安定） |

#### バージョン間比較

| Version | 平均誤差 | vs V1 | vs V2 |
|---------|---------|-------|-------|
| V1 (ESN only) | 0.901m | - | - |
| V2 (Kalman Hybrid) | 0.753m | +16.4% | - |
| V3 (Adaptive) | 0.623m | +30.8% | +17.2% |

#### 複雑軌跡 vs 直線軌跡 比較

軌跡複雑度スコアにより歩行者を選別して比較：

| 手法 | 複雑軌跡 (m) | 直線軌跡 (m) |
|------|-------------|-------------|
| **Kalman** | **1.075** | **1.107** |
| V3 (Adaptive) | 1.523 | 1.188 |
| V2 (ESN+Kalman) | 1.903 | 1.309 |
| V1 (ESN) | 2.375 | 1.477 |

- 複雑軌跡: ped_ids 68, 90, 165, 399, 116（最大角度150°以上）
- 直線軌跡: ped_ids 280, 248, 249, 273, 87（最大角度30°未満）

#### 現状の知見

1. **カルマンフィルタ単体が最も良い結果**
   - 複雑軌跡でも直線軌跡でもKalmanが優位
   - ETHの「複雑」軌跡でもKalmanの速度ベース予測で対応可能

2. **V3適応型ESNはV1比30.8%改善**
   - 軌跡複雑度分析による動的重み調整
   - 直線軌跡ではKalmanとの差が7.3%まで縮小

3. **ESNが効果を発揮する条件**（今後の検証課題）
   - より急激な方向転換（停止→逆方向など）
   - 予測不能な動き（障害物回避、他者との相互作用）
   - 非定常な速度変化

### 評価ツール

```bash
# V1 vs V2 比較
python3 tools/eth_v1_v2_comparison.py --ped_ids 399 168 269 177 178

# V3 適応型ESN評価
python3 tools/eth_v3_adaptive.py --ped_ids 399 168 269 177 178

# 従来手法との比較
python3 tools/eth_method_comparison.py --ped_ids 399 168 269 177 178
```

## ドキュメント

詳細は [docs/](docs/) を参照してください。

- [path_prediction概要](docs/path_prediction.md) - システム概要
- [leg_finder詳細](docs/path_prediction_leg_finder.md) - 脚検出アルゴリズム
- [ESN経路予測詳細](docs/path_prediction_esn.md) - ESNアルゴリズム
- [デプロイメント](docs/path_prediction_deployment.md) - Docker設定
- [検証ツール](docs/path_prediction_tools.md) - ETHデータセット評価スクリプト
- [ETHデータセット評価](docs/path_prediction_eth_evaluation.md) - 予測精度の検証結果（V1/V2/V3比較含む）
- [V2改良検証](docs/path_prediction_v2_improvements.md) - カルマンハイブリッドの検証記録
