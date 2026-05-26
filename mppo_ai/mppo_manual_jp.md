# Fafnir MPPO AI 開発・強化学習マニュアル

このドキュメントは、Fafnirの強化学習モデル（Maskable PPO）の学習、評価、および対戦ボットを管理するプログラム群の仕様書です。

---

## 1. ファイル構成とそれぞれの目的

`mppo_ai` 内のプログラムは、目的ごとにレイヤーに分かれています。

### 1.1. ゲームエンジン・レイヤー（環境）
*   **`mppo_ai/rl/fast_engine.py`**
    *   **目的:** 学習を限界まで高速化するためのコア・ゲームロジック。
    *   **特徴:** Numba (`@njit`) を使用してC言語レベルの速度にコンパイル。
*   **`mppo_ai/rl/game_env_fast.py`**
    *   **目的:** `fast_engine.py` をSB3が扱える形（Gymnasium環境）に包むラッパー。`train.py` 専用。
*   **`mppo_ai/rl/game_env.py`**
    *   **目的:** Numbaを使わない、純粋なPython実装のGymnasium環境。`evaluate.py` 専用。

### 1.2. 学習・評価スクリプト・レイヤー
*   **`mppo_ai/rl/train.py`**
    *   **目的:** モデルを学習させるメインスクリプト（自己対戦カリキュラム）。
*   **`mppo_ai/rl/evaluate.py`**
    *   **目的:** 学習済みモデルをテスト用ボットと戦わせて勝率を計測するスクリプト。

### 1.3. クライアント・レイヤー（オンライン対戦）
*   **`mppo_ai/clients/rl_bot.py` 等**
    *   **目的:** サーバー (`server_0424.py`) に接続し、対戦するプログラム。

---

## 2. 仕様詳細：AIの認識と報酬

### 2.1. 観測空間 (Observation Space - 34次元 / 旧36次元)
※ 最新版(v3)は34次元、旧モデル(v2)は36次元です。
1.  `0-5`: **自分の手札** (6色)
2.  `6-11`: **現在のオファー** (6色)
3.  `12-17`: **ゴミ箱の石** (6色)
4.  `18-23`: **相手の確定手札** (6色)
5.  `24`: **相手の不確定な手札の枚数** (1次元)
6.  `25-30`: **自分の確定手札** (6色)
7.  `31`: **袋の残り枚数**
8.  `32`: **自分が管理人(Caretaker)かどうか**
9.  `33`: **自分の手札の期待値スコア**

*(※ v2では、インデックス31に「自分のスコア」、32に「相手のスコア」が含まれます。)*

### 2.2. 行動空間 (Action Space)
*   **`MultiDiscrete([11] * 6)`**: 6つの色それぞれについて「0〜10個」のどれを出すか同時に決定します。

### 2.3. 報酬設計 (Reward Design)
1.  **ラウンド終了時の報酬:** `(自分が得たスコア - 相手が得たスコア) ÷ 30`
2.  **オークションのスコア差分報酬:** `(自分のスコア増加 - 相手のスコア増加) × 0.02`
3.  **期待値の増減報酬:** `(行動後の期待値 - 行動前の期待値) × 0.005`
4.  **行動ペナルティ:** 入札した石の数の「2乗 × -0.002」の罰則（無駄捨て防止）。

---

## 3. コマンドライン引数と使用方法

### 3.1. 学習 (`mppo_ai/rl/train.py`)
```bash
python mppo_ai/rl/train.py [オプション]
```
| 引数 | デフォルト | 説明 |
| :--- | :--- | :--- |
| `--total-steps` | `500000` | 学習の総ステップ数。 |
| `--n-envs` | `8` | 並列実行する環境数。 |
| `--device` | `auto` | `cpu` または `cuda`。 |
| `--resume` | `None` | 学習を再開したい `.zip` モデルのパス。 |

**TensorBoardでの確認:**
```bash
venv\Scripts\python.exe -m tensorboard.main --logdir mppo_ai/rl/output/logs
```

### 3.2. 評価 (`mppo_ai/rl/evaluate.py`)
```bash
python mppo_ai/rl/evaluate.py --model mppo_ai/rl/output/fafnir_final.zip [オプション]
```
| 引数 | デフォルト | 説明 |
| :--- | :--- | :--- |
| `--model` | (必須) | 評価したいモデルのパス。 |
| `--games` | `200` | 各ボットとの対戦回数。 |
| `--deterministic` | 指定なし | 確率的に揺らがない最強の手を選ぶ。 |

### 3.3. クライアント起動

#### ① RLボット (`mppo_ai/clients/rl_bot_v3.py`)
```bash
python mppo_ai/clients/rl_bot_v3.py --model mppo_ai/rl/output/fafnir_final.zip --room room1
```
| 引数 | デフォルト | 説明 |
| :--- | :--- | :--- |
| `--model` | (必須) | 使用するモデルのパス。 |
| `--deterministic` | `1` | `1`なら最強の手、`0`なら少し揺らぎを持たせる。 |

#### ② PIMC探索ボット (`mppo_ai/clients/pimc_bot_v3.py`)
実行時にモンテカルロ探索を行ってより強力な手を打つボットです。
```bash
python mppo_ai/clients/pimc_bot_v3.py --model mppo_ai/rl/output/fafnir_final.zip --room room1
```
| 引数 | デフォルト | 説明 |
| :--- | :--- | :--- |
| `--search-time` | `0.2` | 1手あたりの探索に使う時間（秒）。 |
| `--candidates` | `4` | 1決定化パターンごとの候補手数。 |
| `--det-batch` | `4` | 1GPU評価にまとめる決定化パターン数。 |
