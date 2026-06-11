# Deep CFR AI for Fafnir

`cfr_ai` は、`server_0424.py` 上の Fafnir をプレイする Deep CFR 系 AI です。
学習、評価、checkpoint リーグ、Socket.IO 対戦 bot を含みます。

## 現在の設計方針

対戦時の行動選択では、公開情報と自分の手札だけを使います。
サーバー payload に相手の未公開情報が含まれていても、`clients/cfr_bot.py` の `fair_state_view()` で除外します。

使う情報:

- 自分の手札
- 現在の offer
- trash
- caretaker
- 解決済みの `last_result`
- 解決済み bid から推定した confirmed hand

使わない情報:

- `BIDDING` 中の相手の未解決 bid
- `action_log` に残る相手手札や未解決 bid
- `resolve_before` / `resolve_after` に含まれる全員の手札
- 相手の `last_bid`
- 相手が先に提出済みかどうかを利用した後手 best response
- bag 残数
- 自分と相手のスコア
- round / turn
- offer 合計枚数
- 自分と相手の手札合計枚数
- trash 合計枚数

## ディレクトリ構成

```text
cfr_ai/
├── cfr_manual_jp.md
├── clients/
│   └── cfr_bot.py         # server_0424.py 接続用 bot
└── ai/
    ├── action_space.py    # bid 行動の列挙、合法手マスク
    ├── evaluate.py        # checkpoint 評価 CLI
    ├── fairness_check.py  # 非公開情報を使わないことの検査
    ├── game_engine.py     # 学習・評価用ゲームエンジン。Numba 高速化もここに統合済み
    ├── league.py          # checkpoint リーグ評価と採用候補ランキング
    ├── networks.py        # Regret / Strategy / Value Network
    ├── observation.py     # 33 次元観測と BidTracker
    ├── parallel.py        # 並列 traversal worker
    ├── symmetry.py        # 色置換によるデータ拡張
    ├── train.py           # 学習 CLI
    └── trainer.py         # Deep CFR 本体
```

`fast_engine.py` は旧高速エンジンです。現行の高速化は `game_engine.py` に統合済みで、現行コードからの import 参照はありません。

## 学習方法

基本:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.train
```

途中から再開:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.train --resume
```

推奨の標準起動:

```bash
uv --cache-dir .uv-cache run python -u -m cfr_ai.ai.train --resume --workers 7 --iterations 4000 --traversals 2000 --train-steps 100 --target-mode terminal --past-opponent-prob 0.15
```

短い動作確認:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.train --iterations 1 --traversals 2 --train-steps 1 --workers 1 --eval-every 0 --final-eval-games 0 --archive-every 0 --save-dir .codex_tmp/cfr_smoke
```

## 主な学習オプション

| オプション | デフォルト | 説明 |
|---|---:|---|
| `--iterations` | `4000` | CFR の反復回数 |
| `--traversals` | `2000` | 1 iteration あたりの traversal 数。1 traversal は原則 1 round 分の自己対戦 |
| `--hidden` | `256` | 各ネットワークの隠れ層サイズ |
| `--lr` | `5e-4` | 学習率 |
| `--batch-size` | `2048` | NN 学習バッチサイズ |
| `--train-steps` | `100` | 1 iteration あたりの NN 更新回数 |
| `--max-depth` | `50` | 1 traversal の最大ターン数。通常 round 終了まで進める |
| `--augments` | `3` | 色置換データ拡張数 |
| `--buffer-capacity` | `1000000` | 各 replay buffer の最大サンプル数 |
| `--save-dir` | `cfr_ai/ai/checkpoints` | checkpoint 保存先 |
| `--save-every` | `100` | 通常 checkpoint の保存間隔 |
| `--archive-every` | `100` | 履歴 checkpoint の保存間隔。`0` で無効 |
| `--eval-every` | `0` | 学習中の簡易評価間隔。`0` で無効 |
| `--final-eval-games` | `0` | 学習終了時の評価ゲーム数。`0` で無効 |
| `--target-mode` | `terminal` | `terminal` は round 終端報酬のみ。`dense` は各 decision point からの差分報酬 |
| `--score-rand` | 無効 | 隠れスコア乱数化を有効化する。現在はスコアを観測に入れないため通常は使わない |
| `--epsilon` | `0.3` | 初期探索率。学習進行で decay |
| `--workers` | `7` | 並列 worker 数。`1` は単一 process |
| `--past-opponent-prob` | `0.15` | non-traverser に過去の凍結 Regret Network を使う確率 |
| `--max-past-opponents` | `8` | メモリ上に保持する過去方策数 |
| `--past-opponent-selection` | `recent` | `recent` / `spread` / `random` / `manifest` |
| `--past-opponent-manifest` | なし | manifest 選択時に使う checkpoint 一覧 |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |

## iterations / traversals / steps

- `iterations`: CFR の大きな反復単位。各 iteration で traversal を集め、ネットワークを更新します。
- `traversals`: 自己対戦サンプル数。現行実装では 1 traversal は 1 round を基本単位として進みます。
- `train-steps`: その iteration で Regret / Strategy / Value Network を何 batch 更新するかです。
- `steps`: MPPO では環境 step を指しますが、CFR では主に NN 更新回数や内部 turn 数を指すため、文脈で意味が変わります。

## 学習アルゴリズム

現行実装は Outcome Sampling MCCFR と Deep CFR の組み合わせです。

1. `new_game()` で自己対戦状態を生成する。
2. 各 decision point で 33 次元 observation と合法手 mask を作る。
3. Regret Network の出力から regret matching で方策を作る。
4. traverser は epsilon 探索を混ぜて行動を sample する。
5. opponent は現方策、または一定確率で過去 checkpoint の Regret Network から行動する。
6. round 終了または `max-depth` まで進める。
7. `target-mode=terminal` では round 全体の終端スコア差を全 decision point の target にする。
8. `target-mode=dense` では各 decision point 以降に得たスコア差を target にする。
9. Regret / Strategy / Value buffer にサンプルを追加し、各ネットワークを更新する。

デフォルトは `target-mode=terminal` です。これは金だけを集めるような dense reward の偏りを避けるためです。

## 報酬とスコア

学習時の round 終端価値は次の差分です。

```text
(traverser の round 後スコア - 初期スコア)
- (opponent の round 後スコア - 初期スコア)
```

round が未終了で `max-depth` に達した場合は、現在の手札に対して `compute_hand_score()` を追加して近似します。
`compute_hand_score()` は学習時のみ両者の手札を使って色順位を決めます。これは round 終端の真のスコア計算です。

一方、観測の index 32 は対戦時との整合性を保つため、自分の手札だけから計算する visible hand potential です。

## 高速化設計

`game_engine.py` に Numba JIT 高速化を統合しています。

- `FafnirState` の `hand`, `bag`, `trash`, `offer`, `scores` は `numpy.ndarray(int32)`。
- `step_auction()` は `_step_auction_fast()` に接続され、auction resolve、round end 判定、offer setup、round reset まで JIT 側で処理します。
- `build_observation()` は `build_observation_fast_arrays()` を使います。
- `get_legal_mask()` は `get_legal_mask_fast()` を使います。
- `BidTracker.update_from_auction()` は `update_confirmed_fast()` を使います。
- `train.py` と `parallel.py` は `warmup()` を呼び、初回 JIT compile の遅延を学習本体から外します。

`state.bid_history` は互換のため属性として残っていますが、現行の学習・評価・クライアントからは参照されていません。
高速化のため、`step_auction()` 経由では `bid_history` を更新しません。

## 観測空間

観測は 33 次元です。

| Index | 内容 | 次元 |
|---|---|---:|
| 0-5 | 自分の手札 counts | 6 |
| 6-11 | 現在の offer counts | 6 |
| 12-17 | trash counts | 6 |
| 18-23 | 相手の confirmed hand | 6 |
| 24 | 相手の未確定手札枚数 | 1 |
| 25-30 | 自分の confirmed hand | 6 |
| 31 | 自分が caretaker か | 1 |
| 32 | 自分の手札だけから見た visible hand potential | 1 |

## 行動空間

行動は 6 色の bid counts です。

- 最大合計 bid 数: `6`
- 色ごとの最大 bid 数: `6`
- 行動数: `924`

合法手条件:

- 手札にない石は bid できない。
- offer と同色の石は bid できない。

## ネットワーク

| ネットワーク | 入力 | 出力 | 役割 |
|---|---:|---:|---|
| Regret Network | 33 | 924 | regret を推定し、regret matching に使う |
| Strategy Network | 33 | 924 | 推論用の平均方策 |
| Value Network | 33 | 1 | decision point の value baseline |

Regret / Strategy Network は Dueling Architecture です。

```text
Input (33)
  -> Backbone
  -> Value Stream      -> V(s)
  -> Advantage Stream  -> A(s,a)
  -> Output = V(s) + A(s,a) - mean(A(s,*))
```

## 評価

基本評価:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 500
```

temperature 比較:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 500 --temperatures 0.05,0.1,0.2,0.3,0.5
```

Regret 方策比較:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 500 --policy regret
```

heuristic 相手:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 500 --opponent heuristic
```

金 bid 率の確認:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 1000 --opponent heuristic --temperature 0.3 --gold-report
```

full game 評価:

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.evaluate --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt --games 100 --full-game
```

## checkpoint リーグ評価

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.league --checkpoint-glob "cfr_ai/ai/checkpoints/**/*.pt" --games 1000 --league-games 300 --temperatures 0.1,0.2,0.3 --policies strategy,regret --opponents heuristic --gold-penalty 20
```

出力先:

- `baseline_results.csv`: random/heuristic など固定相手への評価
- `league_results.csv`: 上位 checkpoint 同士の対戦
- `ranking.csv`: 総合ランキング
- `past_opponents.txt`: `--past-opponent-manifest` に渡せる候補一覧

## 対戦 bot

基本起動:

```bash
uv --cache-dir .uv-cache run python cfr_ai/clients/cfr_bot.py --url http://127.0.0.1:8765 --room room1 --name DeepCFR --checkpoint cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt
```

主なオプション:

| オプション | デフォルト | 説明 |
|---|---:|---|
| `--url` | `http://127.0.0.1:8765` | 接続先 server |
| `--room` | `room1` | 参加 room |
| `--name` | `DeepCFR` | bot 名 |
| `--checkpoint` | `cfr_ai/ai/checkpoints/deep_cfr_checkpoint.pt` | 読み込む checkpoint |
| `--temperature` | `0.3` | 方策 softmax 温度。低いほど決定的 |
| `--policy` | `strategy` | `strategy` または `regret` |
| `--deterministic` | `False` | 最大確率行動を選ぶ |
| `--device` | `cpu` | `cpu` / `cuda` |

`strategy` が Deep CFR の基本運用です。`regret` は短期学習 checkpoint で強いことがありますが、採用前に必ず評価してください。

## フェアプレイ検査

```bash
uv --cache-dir .uv-cache run python -m cfr_ai.ai.fairness_check
```

この検査は、相手手札、未解決 bid、`action_log` などの漏洩情報が変わっても、fair view、観測、選択行動が変わらないことを確認します。

## 運用上の注意

- 最後の checkpoint を自動採用せず、必ず評価で checkpoint と temperature を選んでください。
- `--eval-every` の random 評価は進捗確認用です。採用判断には heuristic、temperature sweep、checkpoint リーグを使ってください。
- 観測、報酬、行動空間、ゲームエンジンを変えた後は、古い checkpoint との強さ比較は参考値扱いにしてください。
- `target-mode=dense` は解析用に残していますが、通常学習は `terminal` を使ってください。
