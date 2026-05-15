# Deep CFR AI for Fafnir

Fafnir（ファフニル）の強いAIを **Deep CFR（Counterfactual Regret Minimization）** で作成するプロジェクトです。

## アルゴリズム概要

**Outcome Sampling MCCFR + Deep Learning** を採用しています。

1. ゲームを最初から最後まで自己対戦（トラバーサル）し、各意思決定ポイントで「この手を選ばなかったことへの後悔値（Regret）」を推定
2. Regret をニューラルネットワーク（Regret Network）で近似
3. CFR の反復を通じて得られた戦略の平均を Strategy Network で学習
4. **Strategy Network の出力がナッシュ均衡に近い最終方策** となる

### なぜ Deep CFR か

- ファフニルは**不完全情報・同時着手ゲーム**であり、通常の強化学習（PPO等）ではジャンケンのように特定の相手にだけ強い偏った戦略に陥りやすい
- CFR は理論的にナッシュ均衡に収束するため、**どんな相手にも搾取されにくい**堅牢な戦略を獲得できる

### 計算量削減テクニック

| テクニック | 説明 |
|---|---|
| **Outcome Sampling** | 各トラバーサルで全合法手を展開せず1手だけサンプルし、重要度重みで補正。計算量が O(分岐^深さ) → O(深さ) に削減 |
| **色対称性データ拡張** | 金以外の5色を入れ替えた盤面を追加学習データとして生成し、学習効率を向上 |
| **Value Network** | ゲーム終了まで探索せず、途中の盤面をNNで評価して打ち切り |
| **Epsilon 探索** | 探索中に一定確率でランダム行動を混ぜ、戦略の多様性を維持 |

## ファイル構成

```
cfr_ai/
├── README.md              # このファイル
├── clients/
│   └── cfr_bot.py         # サーバー接続用の対戦ボット
└── ai/
    ├── __init__.py
    ├── game_engine.py     # スタンドアロンのゲームエンジン（サーバー不要）
    ├── action_space.py    # 行動空間の列挙・マスキング（54,264アクション）
    ├── observation.py     # 34次元観測ベクトルの構築 + BidTracker
    ├── networks.py        # NN定義（Regret / Strategy / Value）
    ├── symmetry.py        # 色対称性によるデータ拡張
    ├── trainer.py         # Deep CFR トレーナー本体
    └── train.py           # 学習用CLIスクリプト   
```

## 観測空間（34次元）

| Index | 内容 | 次元 |
|---|---|---|
| 0-5 | 自分の手札（6色） | 6 |
| 6-11 | 現在のオファー（6色） | 6 |
| 12-17 | ゴミ箱の石（6色） | 6 |
| 18-23 | 相手の確定手札（6色）※入札履歴から推測 | 6 |
| 24 | 相手の不確定な手札の枚数 | 1 |
| 25-30 | 自分の確定手札（6色）※相手にバレている石 | 6 |
| 31 | 袋の残り枚数（正規化） | 1 |
| 32 | 自分が管理人(Caretaker)か | 1 |
| 33 | 自分の手札の期待値スコア（正規化） | 1 |

## 行動空間（54,264アクション）

- 6色 × 各色0〜15個 × 合計0〜15個の全組み合わせを列挙
- 毎ターン「手札にない色」「オファーと同色」をマスクし、合法手のみに制限
- 典型的な合法手数: 20〜60（手札が多いと数百）
- **ルール上の行動制限は一切なし**。手札15枚超の場合のみクリップ処理

## 使い方

### 前提条件

```bash
# project ディレクトリで仮想環境を有効化
.\venv\Scripts\activate

# PyTorch が必要（CPU版でも動作可能）
pip install torch
```

### 学習

```bash
python -m cfr_ai.ai.train --iterations 100 --traversals 500 --hidden 256
```

主なオプション:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--iterations` | 100 | CFR反復回数 |
| `--traversals` | 200 | 1反復あたりの自己対戦回数 |
| `--hidden` | 256 | NN隠れ層の次元数 |
| `--lr` | 0.001 | 学習率 |
| `--batch-size` | 2048 | 学習バッチサイズ |
| `--max-depth` | 30 | 1トラバーサルの最大ターン数 |
| `--augments` | 3 | 色対称性の拡張数 |
| `--save-dir` | `cfr_ai/ai/checkpoints` | チェックポイント保存先 |
| `--resume` | - | 前回の学習から再開 |
| `--save-every` | 10 | N反復ごとに保存 |

### 学習時間の目安（CPU）

| 設定 | 時間 |
|---|---|
| 50 traversals × 1 iter | 約50秒 |
| 500 traversals × 100 iter | 約12〜15時間 |
| 1000 traversals × 200 iter | 約2〜3日 |

> GPU（CUDA）があれば NN の学習部分が高速化されます

### 対戦ボットの起動

学習済みモデルを使ってサーバー上で対戦:

```bash
python cfr_ai\ai\cfr_bot.py ^
    --url http://127.0.0.1:8765 ^
    --room room1 ^
    --name DeepCFR ^
    --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt ^
    --temperature 0.3
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--url` | `http://127.0.0.1:8765` | サーバーURL |
| `--room` | `room1` | ルームID |
| `--name` | `DeepCFR` | プレイヤー名 |
| `--checkpoint` | `cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt` | モデルファイル |
| `--temperature` | 0.3 | 行動のランダム性（低い=決定論的） |

> チェックポイントが無い場合はランダムプレイにフォールバック

### サンプルAIとの対戦

```bash
# ターミナル1: サーバー起動
uvicorn server_0424:socket_app --host 0.0.0.0 --port 8765

# ターミナル2: Deep CFR ボット
python cfr_ai\ai\cfr_bot.py --url http://127.0.0.1:8765 --room room1 --name DeepCFR

# ターミナル3: サンプルAI（ランダム）
python clients\ai_bot_sample.py --url http://127.0.0.1:8765 --room room1 --name RandomAI
```

## ネットワーク構成

3つの独立したネットワーク（Residual MLP + LayerNorm）:

| ネットワーク | 入力 → 出力 | 役割 |
|---|---|---|
| **Regret Network** | 34 → 54,264 | CFR探索中の戦略計算（Regret Matching） |
| **Strategy Network** | 34 → 54,264 | 推論時の最終方策（ナッシュ均衡近似）|
| **Value Network** | 34 → 1 | 途中盤面の勝率推定（探索打ち切り用）|
