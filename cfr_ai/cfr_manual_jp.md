# Deep CFR AI for Fafnir (v2)

Fafnir（ファフニル）の強いAIを **Deep CFR（Counterfactual Regret Minimization）** で作成するプロジェクトです。

## v2 での主な改善点

| 改善 | 内容 |
|---|---|
| **Regret推定の修正** | 全合法アクションの反事実後悔を正しく計算（v1は選択アクションのみ） |
| **観測空間拡張** | 34次元 → 42次元（スコア、ラウンド/ターン番号、手札サイズ等を追加） |
| **Dueling Architecture** | Value stream + Advantage stream に分離し、大行動空間での学習効率を向上 |
| **DCFR重み付け** | 古いデータを割り引き、新しいイテレーションを重視する Discounted CFR |
| **適応的探索** | ε=0.4 から始まり 0.997^t で自動減衰（最小0.05） |
| **スコアランダム化** | 50%のゲームでランダムな開始スコアを設定し、スコア状況に応じた戦略を学習 |
| **評価パイプライン** | vs Random / vs 過去チェックポイント の勝率を自動測定 |
| **推論時サーチ** | Monte Carlo ロールアウトによる行動選択の改善（オプション） |

## アルゴリズム概要

**Outcome Sampling MCCFR + Deep Learning** を採用しています。

1. ゲームを最初から最後まで自己対戦（トラバーサル）し、各意思決定ポイントで「この手を選ばなかったことへの後悔値（Regret）」を推定
2. Regret をニューラルネットワーク（Regret Network）で近似
3. CFR の反復を通じて得られた戦略の平均を Strategy Network で学習
4. **Strategy Network の出力がナッシュ均衡に近い最終方策** となる

### なぜ Deep CFR か

- ファフニルは**不完全情報・同時着手ゲーム**であり、通常の強化学習（PPO等）ではジャンケンのように特定の相手にだけ強い偏った戦略に陥りやすい
- CFR は理論的にナッシュ均衡に収束するため、**どんな相手にも搾取されにくい**堅牢な戦略を獲得できる

### v2 の計算量削減テクニック

| テクニック | 説明 |
|---|---|
| **Outcome Sampling** | 各トラバーサルで全合法手を展開せず1手だけサンプルし、重要度重みで補正。O(深さ) |
| **色対称性データ拡張** | 金以外の5色を入れ替えた盤面を追加学習データとして生成 |
| **Dueling Architecture** | State value と Action advantage を分離し、行動空間が大きくても効率的に学習 |
| **DCFR** | 古いイテレーションのデータを割り引き、収束を加速 |
| **スコアランダム化** | マルチラウンド探索の代替。ゲーム全体の状況を0コストで学習 |
| **Adaptive ε** | 序盤は高探索（ε=0.4）、収束に伴い自動的に減少 |

## ファイル構成

```
cfr_ai/
├── cfr_manual_jp.md       # このファイル
├── clients/
│   └── cfr_bot.py         # サーバー接続用の対戦ボット
└── ai/
    ├── __init__.py
    ├── game_engine.py     # スタンドアロンのゲームエンジン
    ├── action_space.py    # 行動空間の列挙・マスキング（924アクション）
    ├── observation.py     # 42次元観測ベクトルの構築 + BidTracker
    ├── networks.py        # NN定義（Dueling Regret / Strategy / Value）
    ├── symmetry.py        # 色対称性によるデータ拡張
    ├── trainer.py         # Deep CFR トレーナー本体 (v2)
    ├── train.py           # 学習用CLIスクリプト
    ├── evaluate.py        # 評価パイプライン（vs Random / vs Checkpoint）
    └── search.py          # 推論時リアルタイムサーチ
```

## 観測空間（42次元）

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
| 34 | 自分のゲームスコア（/1000で正規化） | 1 |
| 35 | 相手のゲームスコア（/1000で正規化） | 1 |
| 36 | ラウンド番号（/20で正規化） | 1 |
| 37 | ターン番号（/30で正規化） | 1 |
| 38 | オファーの合計枚数（/10で正規化） | 1 |
| 39 | 自分の手札合計（/20で正規化） | 1 |
| 40 | 相手の手札合計（/20で正規化） | 1 |
| 41 | ゴミ箱の合計枚数（正規化） | 1 |

## 行動空間（924アクション）

- 6色 × 各色0〜6個 × 合計0〜6個の全組み合わせを列挙
- 毎ターン「手札にない色」「オファーと同色」をマスクし、合法手のみに制限
- 典型的な合法手数: 20〜60（手札が多いと数百）

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
# デフォルト設定でそのまま実行可能
python -m cfr_ai.ai.train
```

> `Ctrl+C` で中断しても、自動的にチェックポイントが保存されます。
> 再開するには `--resume` を追加してください。

主なオプション:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--iterations` | 1000 | CFR反復回数 |
| `--traversals` | 1000 | 1反復あたりの自己対戦回数 |
| `--hidden` | 192 | NN隠れ層の次元数 |
| `--lr` | 0.001 | 学習率 |
| `--batch-size` | 2048 | 学習バッチサイズ |
| `--train-steps` | 100 | 1反復あたりのNN学習ステップ数 |
| `--max-depth` | 50 | 1トラバーサルの最大ターン数 |
| `--augments` | 1 | 色対称性の拡張数 |
| `--save-dir` | `cfr_ai/ai/checkpoints` | チェックポイント保存先 |
| `--resume` | - | 前回の学習から再開 |
| `--save-every` | 20 | N反復ごとに保存 |
| `--device` | auto | cpu/cuda/auto |
| `--workers` | 1 | 並列ワーカー数 (0=auto) |
| `--eval-every` | 50 | N反復ごとにvsRandom評価 (0=無効) |
| `--no-score-rand` | - | スコアランダム化を無効化 |

### 学習時間の目安（Ryzen 7 5800HS, hidden=192）

| 設定 | 時間（推定） |
|---|---|
| 200 traversals × 50 iter | 約3〜5時間 |
| 500 traversals × 100 iter | 約12〜20時間 |
| 1000 traversals × 200 iter | 約2〜4日 |

> `--workers 0` を指定すると自動で並列ワーカー数を設定します

### 評価

```bash
# vs Randomの勝率を測定
python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 500

# vs 過去チェックポイント
python -m cfr_ai.ai.evaluate --checkpoint new_model.pt --vs-checkpoint old_model.pt --games 500
```

### 対戦ボットの起動

学習済みモデルを使ってサーバー上で対戦:

```bash
python cfr_ai\clients\cfr_bot.py ^
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

## ネットワーク構成（v2: Dueling Architecture）

3つの独立したネットワーク:

| ネットワーク | 入力 → 出力 | 構造 | 役割 |
|---|---|---|---|
| **Regret Network** | 42 → 924 | Dueling (Value+Advantage) | CFR探索中の戦略計算 |
| **Strategy Network** | 42 → 924 | Dueling (Value+Advantage) | 推論時の最終方策 |
| **Value Network** | 42 → 1 | ResBlock MLP | 途中盤面の勝率推定 |

### Dueling Architecture

```
Input (42-dim) → Backbone (ResBlock×2) → ┬─ Value Stream  → V(s)    [1-dim]
                                         └─ Advantage Stream → A(s,a) [924-dim]

Output: Q(s,a) = V(s) + A(s,a) - mean(A(s,.))
```

状態の「良さ」と各行動の「相対的な良さ」を分離して学習するため、
行動空間が大きい（924アクション）場合でも効率的に学習できます。
