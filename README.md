# Fafnir AI

A reinforcement learning AI for the board game **Fafnir**, built with [MaskablePPO](https://sb3-contrib.readthedocs.io/) (Stable-Baselines3) and a Numba-accelerated game engine.

The project includes a full training pipeline, evaluation tools, and multiple client programs for online play — including a **PIMC (Perfect Information Monte Carlo) search bot** that combines the trained neural network with real-time lookahead search.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [AI Specification](#ai-specification)
  - [Observation Space](#observation-space-36-dimensions)
  - [Action Space](#action-space)
  - [Reward Design](#reward-design)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Game Server](#game-server)
  - [Clients](#clients)
- [PIMC Search Algorithm](#pimc-search-algorithm)

---

## Project Structure

The codebase is organized into three layers by purpose:

### Game Engine Layer

| File | Purpose |
|---|---|
| `mppo_ai/rl/fast_engine.py` | Core game logic compiled to native speed via Numba `@njit`. Handles observation computation, hand-value estimation, and opponent known-hand tracking. |
| `mppo_ai/rl/game_env_fast.py` | Gymnasium wrapper around `fast_engine.py`. Used exclusively by `train.py`. |
| `mppo_ai/rl/game_env.py` | Pure-Python Gymnasium environment (no Numba). Used by `evaluate.py`. Also contains built-in test opponents (Random, Greedy, etc.). |

### Training & Evaluation Layer

| File | Purpose |
|---|---|
| `mppo_ai/rl/train.py` | Main training script with self-play curriculum. |
| `mppo_ai/rl/evaluate.py` | Benchmarks a trained model against test bots and reports win rates. |

### Client Layer (Online Play)

| File | Purpose |
|---|---|
| `mppo_ai/clients/rl_bot.py` | Connects a trained RL model to the game server for online play. Reconstructs the same 36-dim observation vector client-side, including opponent hand tracking. |
| `mppo_ai/clients/pimc_bot.py` | Enhanced RL client with PIMC lookahead search. Inference-only — does **not** train the model. |
| `mppo_ai/clients/human_cli.py` | Terminal-based client for human players. |
| `mppo_ai/clients/ai_bot_sample.py` | Simple rule-based / random bot. Useful as a test opponent. |
| `mppo_ai/clients/spectator_gui.py` | Pygame desktop spectator GUI. Displays public game state graphically. |
| `mppo_ai/clients/web_gui/` | Browser-based UI (HTML/CSS/JS). Open `index.html` to play via the game server. |

---

## Getting Started

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt
```

---

## AI Specification

### Observation Space (34 dimensions / legacy 36 dimensions)

All values are normalized to the range `[0.0, 1.0]`.
*Note: The latest architecture (v3) omits absolute scores to train on isolated rounds (34-dim). The legacy architecture (v2) includes scores (36-dim).*

| Index | Feature | Description |
|---|---|---|
| 0–5 | **My hand** (6 colors) | Stone counts in my hand, normalized by initial bag count per color. |
| 6–11 | **Current offer** (6 colors) | Stones on offer this auction, normalized by 10. |
| 12–17 | **Trash** (6 colors) | Stones in the trash pile, normalized by the trash limit. |
| 18–23 | **Opponent's confirmed hand** (6 colors) | Minimum stones the opponent is known to hold, inferred from auction history. |
| 24 | **Opponent's unknown count** | Number of opponent stones whose color is unknown (total hand − confirmed). Normalized by 15. |
| 25–30 | **My confirmed hand** (6 colors) | What the opponent can deduce about my hand from past auctions. |
| 31 | **Bag remaining** | Total stones left in the bag, normalized by initial bag total. |
| 32 | **Am I the caretaker?** | `1.0` if I am the current caretaker, `0.0` otherwise. |
| 33 | **Hand potential score** | Expected round-end score if the round ended now. Normalized from range [−15, +60]. |

*(In the legacy 36-dim model, Index 31 is My Score, and Index 32 is Opponent's Score).*

### Action Space

**`MultiDiscrete([11] × 6)`** — For each of the 6 stone colors, the agent simultaneously selects how many to bid (0–10). An action mask enforces legality: colors present in the offer are forced to 0, and bids cannot exceed the player's hand count.

### Reward Design

The reward signal is composed of four terms:

| Component | Formula | Purpose |
|---|---|---|
| **Round Score** | `(round_score_diff) / 30.0` | Ultimate objective. A game episode is exactly one round. |
| **Score delta** | `(my_score_change − opp_score_change) × 0.02` | Per-turn progress signal. |
| **Potential shaping** | `(potential_after − potential_before) × 0.005` | Rewards improving hand composition (acquiring valuable stones, discarding bad ones). |
| **Action tax** | `(stones_bid^2) × −0.002` | Quadratic penalty. 1-2 stones is cheap, 4+ stones is heavily penalized to discourage wasteful discarding. |

---

## Usage

### Training

```bash
python mppo_ai/rl/train.py [options]
```

| Argument | Default | Description |
|---|---|---|
| `--total-steps` | `500000` | Total training timesteps. |
| `--score-to-win` | `40` | Score required to win a game. |
| `--max-turns` | `500` | Maximum turns per game (truncation limit). |
| `--n-envs` | `8` | Number of parallel environments. Match to CPU core count for best speed. |
| `--update-freq` | `100000` | Steps between self-play opponent updates. |
| `--save-freq` | `500000` | Steps between checkpoint saves. |
| `--save-dir` | `mppo_ai/rl/output` | Output directory for models and logs. |
| `--device` | `auto` | `cpu` or `cuda` (GPU). |
| `--resume` | `None` | Path to a `.zip` model to resume training from. |

**Example:**
```bash
python mppo_ai/rl/train.py --total-steps 1000000 --device cuda --n-envs 16 --resume mppo_ai/rl/output/fafnir_final.zip
```

**Monitoring with TensorBoard:**
```bash
# In a separate terminal (with the venv activated):
python -m tensorboard.main --logdir mppo_ai/rl/output/logs
```
Then open `http://localhost:6006/` in your browser.

### Evaluation

Benchmarks a trained model against three test bots (Random, Greedy, Aggressive).

```bash
python mppo_ai/rl/evaluate.py --model mppo_ai/rl/output/fafnir_final.zip [options]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | **(required)** | Path to the model to evaluate. |
| `--games` | `200` | Number of games per opponent type. |
| `--score-to-win` | `40` | Score required to win. |
| `--max-turns` | `500` | Maximum turns per game. |
| `--deterministic` | *(flag)* | If set, the AI always picks the highest-probability action (no exploration noise). |

### Game Server

Start the server before connecting any clients or GUIs:

```bash
uvicorn server_0424:socket_app --host 0.0.0.0 --port 8765
```

Start the fast server:  
```bash
uvicorn fast_server_0424:socket_app --host 0.0.0.0 --port 8765 --workers 4
```

### Clients

All clients connect to a running game server.

#### RL Bot (`mppo_ai/clients/rl_bot_v2.py` / `rl_bot_v3.py`)

Connects the trained neural network directly — fast, no search overhead.
*(Use `v2` for legacy 36-dim models, `v3` for current 34-dim round-based models).*

```bash
python mppo_ai/clients/rl_bot_v3.py --model mppo_ai/rl/output/fafnir_final.zip [options]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | **(required)** | Path to the trained model. |
| `--url` | `http://127.0.0.1:8765` | Server URL. |
| `--room` | `room1` | Room to join. |
| `--name` | `RLBot` | Display name. |
| `--deterministic` | `1` | `1` = greedy (strongest), `0` = stochastic (slight randomness). |
| `--think-delay` | `0.05` | Delay before submitting a bid (seconds). |

#### PIMC Bot (`mppo_ai/clients/pimc_bot_v2.py` / `pimc_bot_v3.py`)

Enhanced client that wraps the trained model with **PIMC lookahead search** for stronger play. This is an **inference-only** client — it does not train or update the model. See [PIMC Search Algorithm](#pimc-search-algorithm) below.
*(Use `v2` for legacy 36-dim models, `v3` for current 34-dim round-based models).*

```bash
python mppo_ai/clients/pimc_bot_v3.py --model mppo_ai/rl/output/fafnir_final.zip [options]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | **(required)** | Path to the trained model. |
| `--url` | `http://127.0.0.1:8765` | Server URL. |
| `--room` | `room1` | Room to join. |
| `--name` | `PIMC_Bot` | Display name. |
| `--search-time` | `0.2` | Seconds allocated for PIMC search per bid. |
| `--candidates` | `4` | Number of candidate bids sampled from the policy per determinization. |
| `--det-batch` | `4` | Number of determinizations batched per GPU value-network call. Higher = better GPU throughput, but coarser time-limit granularity. |

#### Other Clients

```bash
# Human CLI player
python mppo_ai/clients/human_cli.py --name Player1 --room room1

# Rule-based sample bot (useful as a test opponent)
python mppo_ai/clients/ai_bot_sample.py --name DummyBot --room room1

# Pygame spectator GUI (view-only)
python mppo_ai/clients/spectator_gui.py --room room1

# Browser UI — open mppo_ai/clients/web_gui/index.html in a browser
```

---

## PIMC Search Algorithm

Fafnir is a game of **imperfect information** — the opponent's hand is hidden. The PIMC bot overcomes this by performing the following search at each decision point:

1. **Determinization:** Using the known information (opponent's confirmed stones, bag contents, trash), the bot randomly generates multiple plausible opponent hands. Each sample creates a temporary "perfect information" game state.

2. **Candidate Sampling:** The trained PPO policy network generates a set of candidate bids — one deterministic (greedy) bid plus several stochastic samples — in a single batched forward pass.

3. **1-Step Lookahead:** For each candidate bid × each determinization, the Numba-compiled `fast_engine` simulates one step forward (opponent plays a random bid). The resulting board state is then evaluated by the PPO **value network**.

4. **Best-Action Selection:** The candidate bid with the highest average value across all determinizations is selected and submitted.

```
┌─────────────────────────────────────────────────┐
│              Current Game State                 │
│  (my hand, offer, trash, scores, known info)    │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │  Sample K candidate     │
          │  bids from PPO policy   │
          │  (1 batched GPU call)   │
          └────────────┬────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
  Determinize 1   Determinize 2   Determinize N ...
  (random opp     (random opp     (random opp
   hand sample)    hand sample)    hand sample)
       │               │               │
       ▼               ▼               ▼
  Simulate each   Simulate each   Simulate each
  candidate bid   candidate bid   candidate bid
  via fast_engine via fast_engine via fast_engine
       │               │               │
       └───────────────┼───────────────┘
                       │
          ┌────────────┴────────────┐
          │  Batch-evaluate all     │
          │  post-states with PPO   │
          │  value network          │
          │  (1 GPU call per batch) │
          └────────────┬────────────┘
                       │
                       ▼
              Pick candidate with
              highest average value
```

The search runs within a configurable time budget (`--search-time`), processing determinizations in batches of `--det-batch` for GPU efficiency. More time and more candidates generally produce stronger play.
