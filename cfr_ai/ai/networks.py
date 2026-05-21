"""
Neural networks for Deep CFR (v2).

Architecture improvements:
1. Dueling Architecture for Regret/Strategy networks
   - Separate value stream + advantage stream
   - Better learning efficiency for large action spaces
2. Dropout in ResBlocks for regularization
3. Hidden dim: 192 (balanced for CPU training)

Three networks:
1. Regret Network (Advantage Network): predicts counterfactual regrets
2. Strategy Network (Average Strategy Network): predicts action probabilities
3. Value Network: evaluates non-terminal leaf nodes

All take OBS_DIM-dim observation and output appropriate vectors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from .observation import OBS_DIM


class ResBlock(nn.Module):
    """Residual block with LayerNorm and optional Dropout."""

    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.relu(self.ln1(self.fc1(x)))
        h = self.dropout(h)
        h = self.ln2(self.fc2(h))
        return F.relu(h + residual)


class RegretNetwork(nn.Module):
    """
    Predicts counterfactual regret values for each action.

    Dueling Architecture:
    - Shared feature extraction backbone
    - Value stream: estimates state value (scalar)
    - Advantage stream: estimates per-action advantage
    - Output = Value + (Advantage - mean(Advantage))

    This helps learn which states are generally good/bad independently
    of which specific action is best, improving learning efficiency
    when the action space is large (462 actions).

    Input: OBS_DIM-dim observation
    Output: NUM_ACTIONS-dim regret vector
    """

    def __init__(self, obs_dim: int = OBS_DIM, num_actions: int = 1, hidden: int = 192):
        super().__init__()
        # Shared feature backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            ResBlock(hidden),
            ResBlock(hidden),
        )

        # Value stream (state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Advantage stream (per-action advantage)
        self.advantage_stream = nn.Sequential(
            ResBlock(hidden),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        value = self.value_stream(features)          # [B, 1]
        advantage = self.advantage_stream(features)  # [B, A]
        # Dueling: Q(s,a) = V(s) + A(s,a) - mean(A(s,.))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class StrategyNetwork(nn.Module):
    """
    Predicts average strategy (action probabilities).

    Dueling Architecture (same as RegretNetwork).
    Output is passed through softmax (with masking) at inference time.
    """

    def __init__(self, obs_dim: int = OBS_DIM, num_actions: int = 1, hidden: int = 192):
        super().__init__()
        # Shared feature backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            ResBlock(hidden),
            ResBlock(hidden),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            ResBlock(hidden),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class ValueNetwork(nn.Module):
    """
    Value network for depth-limited search.
    Predicts the expected game outcome from a given state.

    Input: OBS_DIM-dim observation
    Output: scalar value in [-1, 1] (probability of winning minus losing)
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 192):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)

        self.res_block1 = ResBlock(hidden)
        self.res_block2 = ResBlock(hidden)

        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.ln1(self.input_layer(x)))
        h = self.res_block1(h)
        h = self.res_block2(h)
        return self.value_head(h).squeeze(-1)


# ============================================================
# Regret Matching
# ============================================================
def regret_matching(
    regrets: np.ndarray,
    legal_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert regret values to a probability distribution using regret matching.

    1. Clamp negative regrets to 0
    2. Mask illegal actions
    3. Normalize to probability distribution
    4. If all legal regrets are 0 or negative, use uniform over legal actions
    """
    # Clamp to positive
    positive_regrets = np.maximum(regrets, 0.0)

    # Apply legal mask
    positive_regrets *= legal_mask

    total = positive_regrets.sum()
    if total > 0:
        return positive_regrets / total
    else:
        # Uniform over legal actions
        legal_count = legal_mask.sum()
        if legal_count > 0:
            return legal_mask.astype(np.float32) / legal_count
        else:
            # Should never happen - at minimum, pass (zero-bid) is legal
            probs = np.zeros_like(regrets)
            probs[0] = 1.0
            return probs


def masked_softmax(
    logits: np.ndarray,
    legal_mask: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Softmax with masking for the strategy network output.
    """
    masked_logits = logits.copy()
    masked_logits[~legal_mask.astype(bool)] = -1e9
    masked_logits /= max(temperature, 1e-8)

    # Stable softmax
    max_val = masked_logits[legal_mask.astype(bool)].max() if legal_mask.any() else 0
    exp_logits = np.exp(masked_logits - max_val)
    exp_logits[~legal_mask.astype(bool)] = 0

    total = exp_logits.sum()
    if total > 0:
        return exp_logits / total
    else:
        legal_count = legal_mask.sum()
        if legal_count > 0:
            return legal_mask.astype(np.float32) / legal_count
        probs = np.zeros_like(logits)
        probs[0] = 1.0
        return probs
