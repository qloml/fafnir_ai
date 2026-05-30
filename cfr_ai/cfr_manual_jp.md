# Deep CFR AI for Fafnir

Fafnir（ファフニル）を `server_0424.py` 上でプレイするための Deep CFR ベースAIです。

このディレクトリには、学習用ゲームエンジン、Deep CFRトレーナー、評価ツール、対戦用Socket.IO botが含まれます。

## 方針

このAIは `server_0424.py` の実装仕様に従います。ただし、サーバーから届く可能性がある非公開情報や、順番入札による後手有利は使いません。AIの意思決定は、公開情報と自分の手札だけに基づく同時入札ゲームとして扱います。

行動選択で使う情報:

- 自分の手札
- ファフニル上の宝石（offer）
- 捨て札
- 山札残数
- 自分と相手のスコア
- 世話係
- ラウンド番号 / ターン番号
- 解決済みの `last_result`
- 解決済み入札から推定できる確定手札

行動選択で使わない情報:

- `BIDDING` 中に `action_log` へ流れる相手の未解決bid
- `resolve_before` / `resolve_after` などに含まれる全員の手札
- `action_log` に残った相手手札
- 相手の未解決 `last_bid`
- 相手が先に提出済みかどうかを使った後手 best response

実装上は `clients/cfr_bot.py` の `fair_state_view()` が、対戦時の行動選択に使う状態から漏洩情報を除外します。

## クイックスタート

Python実行には `uv` を使います。Windows環境でuv cacheが衝突する場合があるため、以下の例では `--cache-dir .uv-cache` を付けています。通常のuv cacheが問題なく使える環境では省略できます。

```bash
uv --version
```

既存チェックポイントを評価:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 500
```

対戦botを起動:

```bash
uv --cache-dir .uv-cache run python cfr_ai/clients/cfr_bot.py --url http://127.0.0.1:8765 --room room1 --name DeepCFR --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --temperature 0.3
```

フェアプレイ境界を検査:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.fairness_check
```

## ディレクトリ構成

```text
cfr_ai/
├── cfr_manual_jp.md
├── clients/
│   └── cfr_bot.py         # server_0424.py接続用の対戦bot
└── ai/
    ├── action_space.py    # bid行動の列挙と合法手マスク
    ├── evaluate.py        # 評価CLI
    ├── fairness_check.py  # 漏洩情報を使わないことの検査
    ├── fast_engine.py     # numba用の高速エンジン部品
    ├── game_engine.py     # 学習・評価用ゲームエンジン
    ├── networks.py        # Regret / Strategy / Value Network
    ├── observation.py     # 42次元観測とBidTracker
    ├── parallel.py        # 並列traversal worker
    ├── search.py          # 推論時rollout search部品
    ├── symmetry.py        # 色置換によるデータ拡張
    ├── train.py           # 学習CLI
    └── trainer.py         # Deep CFR本体
```

## 対戦bot

### 基本起動

```bash
uv --cache-dir .uv-cache run python cfr_ai/clients/cfr_bot.py --url http://127.0.0.1:8765 --room room1 --name DeepCFR --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt
```

### 主なオプション

| オプション | デフォルト | 説明 |
|---|---:|---|
| `--url` | `http://127.0.0.1:8765` | 接続先サーバー |
| `--room` | `room1` | 参加するルームID |
| `--name` | `DeepCFR` | bot名 |
| `--checkpoint` | `cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt` | 読み込むcheckpoint |
| `--temperature` | `0.3` | `strategy` 方策のsoftmax温度。低いほど決定的 |
| `--policy` | `strategy` | `strategy` または `regret` |
| `--deterministic` | `False` | サンプリングせず最大確率行動を選ぶ |
| `--device` | `cpu` | `cpu` / `cuda` |

`--policy strategy` は Strategy Network の平均方策を使います。Deep CFRの基本運用はこちらです。

`--policy regret` は Regret Network の出力から regret matching で方策を作ります。学習が浅いcheckpointではこちらの方が強い場合があるため、評価してから採用してください。

checkpointが存在しない場合、botはランダムプレイにフォールバックします。

## 学習

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.train
```

中断時はcheckpointが保存されます。再開する場合:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.train --resume
```

### 主な学習オプション

| オプション | デフォルト | 説明 |
|---|---:|---|
| `--iterations` | `10000` | CFR反復回数 |
| `--traversals` | `1000` | 1反復あたりの自己対戦回数 |
| `--hidden` | `256` | 隠れ層次元 |
| `--lr` | `5e-4` | 学習率 |
| `--batch-size` | `2048` | 学習バッチサイズ |
| `--train-steps` | `150` | 1反復あたりのNN更新回数 |
| `--max-depth` | `50` | 1traversalの最大ターン数 |
| `--augments` | `3` | 色対称性データ拡張数 |
| `--buffer-capacity` | `1000000` | 各bufferの最大サンプル数 |
| `--save-dir` | `cfr_ai/ai/checkpoints` | checkpoint保存先 |
| `--save-every` | `20` | N反復ごとに保存 |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--workers` | `0` | 並列worker数。`0` は自動設定 |
| `--eval-every` | `50` | N反復ごとの簡易評価。`0` で無効 |
| `--no-score-rand` | `False` | スコアランダム化を無効化 |
| `--epsilon` | `0.3` | 初期探索率 |

学習後は、最後のcheckpointをそのまま採用せず、評価で最良のcheckpointと温度を選んでください。

## 評価

### 基本評価

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 500
```

評価出力には、勝率だけでなく平均得点差、平均ターン数、平均bid枚数、pass率、金をbidした割合が表示されます。

### 温度比較

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 500 --temperatures 0.05,0.1,0.2,0.3,0.5
```

### Regret方策の比較

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 500 --policy regret
```

### ヒューリスティック相手

Random相手だけでは強さを測りにくいため、簡易ヒューリスティック相手でも評価します。

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 500 --opponent heuristic
```

### 過去checkpointとの比較

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint new_model.pt --vs-checkpoint old_model.pt --games 500
```

### 複数checkpointから選択

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint-glob "cfr_ai/ai/checkpoints/**/*.pt" --games 500 --temperatures 0.1,0.2,0.3
```

`BEST` として出力されるcheckpointと温度の組み合わせを、対戦botで使う候補にします。

### フルゲーム評価

通常評価は1ラウンド単位です。スコア状況を含めた実戦寄りの確認には `--full-game` を使います。

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 100 --full-game
```

## フェアプレイ検査

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.fairness_check
```

この検査は、サーバーから漏洩情報が届いても行動選択に影響しないことを確認します。具体的には、`action_log`、未解決 `last_result`、相手 `hand`、相手 `last_bid`、相手 `bid_submitted` を変えても、行動用view、観測ベクトル、選択行動が変わらないことを検査します。

botや観測生成を変更した後は、この検査を必ず実行してください。

## アルゴリズム

このAIは Outcome Sampling MCCFR とニューラルネットワーク近似を組み合わせた Deep CFR です。

1. 自己対戦でゲーム木をサンプリングする
2. 各意思決定点で、選ばなかった行動への後悔値（regret）を推定する
3. Regret Network が後悔値を近似する
4. CFR反復中に得られた平均戦略を Strategy Network が学習する
5. 推論時は Strategy Network、または Regret Network の regret matching 方策を使う

Fafnirは不完全情報かつ同時入札のゲームです。通常の強化学習では特定相手にだけ強い方策へ寄りやすいため、搾取されにくい平均戦略を得る目的でCFR系の手法を使います。

## 観測空間

観測は42次元です。

| Index | 内容 | 次元 |
|---|---|---:|
| 0-5 | 自分の手札（6色） | 6 |
| 6-11 | 現在のoffer（6色） | 6 |
| 12-17 | 捨て札（6色） | 6 |
| 18-23 | 相手の確定手札（入札履歴から推定） | 6 |
| 24 | 相手の不確定な手札枚数 | 1 |
| 25-30 | 自分の確定手札（相手に知られていると推定される石） | 6 |
| 31 | 山札残数（正規化） | 1 |
| 32 | 自分が世話係か | 1 |
| 33 | 自分の手札だけから見た手札ポテンシャル（正規化） | 1 |
| 34 | 自分のゲームスコア（/1000） | 1 |
| 35 | 相手のゲームスコア（/1000） | 1 |
| 36 | ラウンド番号（/20） | 1 |
| 37 | ターン番号（/30） | 1 |
| 38 | offer合計枚数（/10） | 1 |
| 39 | 自分の手札合計（/20） | 1 |
| 40 | 相手の手札合計（/20） | 1 |
| 41 | 捨て札合計枚数（正規化） | 1 |

Index 33 は、相手の非公開手札を使わず、自分の手札だけから計算します。学習時とサーバー対戦時で同じ情報だけを使うためです。

## 行動空間

行動は「どの宝石をbidするか」を6色の個数ベクトルで表します。

- 最大合計bid数: 6
- 色ごとの最大bid数: 6
- 行動数: 924

毎ターン、次の条件で合法手をマスクします。

- 手札にない宝石はbidできない
- offerと同色の宝石はbidできない

`MAX_TOTAL_BID=6` は実用上の計算量と戦略安定性を優先した設定です。大きなbidを可能にすると行動数と学習負荷が増えます。

## ネットワーク

| ネットワーク | 入力 | 出力 | 役割 |
|---|---:|---:|---|
| Regret Network | 42 | 924 | CFR探索中の後悔値推定 |
| Strategy Network | 42 | 924 | 推論用の平均方策 |
| Value Network | 42 | 1 | 途中局面の価値推定 |

Regret Network と Strategy Network は Dueling Architecture を使います。

```text
Input (42)
  -> Backbone
  -> Value Stream      -> V(s)
  -> Advantage Stream  -> A(s,a)
  -> Output = V(s) + A(s,a) - mean(A(s,*))
```

状態全体の良さと行動ごとの差を分けて学習することで、大きめの行動空間でも学習を安定させます。

## 学習上の注意

- 観測や報酬仕様を変えた後は、古いcheckpointをそのまま強さ比較に使わないでください。観測分布が変わるため、再学習が必要です。
- `--eval-every` のvs Randomは進捗確認用です。採用モデルの判断には、temperature sweep、heuristic相手、過去checkpoint比較を使ってください。
- `--policy regret` は短期学習では強い場合がありますが、Deep CFRの基本は平均戦略である `--policy strategy` です。実戦採用前に必ず評価してください。
- 色対称性データ拡張はサンプル効率を上げますが、非金色には同数順位の優先順があります。強いモデルを作る段階では `--augments` の値も比較対象にしてください。
