"""
Neural networks for Deep CFR.

Two networks:
1. Regret Network (Advantage Network): predicts counterfactual regrets
2. Strategy Network (Average Strategy Network): predicts action probabilities

Both take the 34-dim observation and output NUM_ACTIONS-dim vectors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class RegretNetwork(nn.Module):
    """
    Predicts counterfactual regret values for each action.
    
    Architecture: MLP with residual connections and layer normalization.
    Input: 34-dim observation
    Output: NUM_ACTIONS-dim regret vector
    """

    def __init__(self, obs_dim: int = 34, num_actions: int = 1, hidden: int = 256):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)

        self.res_block1 = ResBlock(hidden)
        self.res_block2 = ResBlock(hidden)
        self.res_block3 = ResBlock(hidden)

        self.output_layer = nn.Linear(hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.ln1(self.input_layer(x)))
        h = self.res_block1(h)
        h = self.res_block2(h)
        h = self.res_block3(h)
        return self.output_layer(h)


class StrategyNetwork(nn.Module):
    """
    Predicts average strategy (action probabilities).
    
    Architecture: MLP with residual connections.
    Output is passed through softmax (with masking) at inference time.
    """

    def __init__(self, obs_dim: int = 34, num_actions: int = 1, hidden: int = 256):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)

        self.res_block1 = ResBlock(hidden)
        self.res_block2 = ResBlock(hidden)
        self.res_block3 = ResBlock(hidden)

        self.output_layer = nn.Linear(hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.ln1(self.input_layer(x)))
        h = self.res_block1(h)
        h = self.res_block2(h)
        h = self.res_block3(h)
        return self.output_layer(h)


class ResBlock(nn.Module):
    """Residual block with LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.relu(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(h))
        return F.relu(h + residual)


class ValueNetwork(nn.Module):
    """
    Value network for depth-limited search.
    Predicts the expected game outcome from a given state.
    
    Input: 34-dim observation
    Output: scalar value in [-1, 1] (probability of winning minus losing)
    """

    def __init__(self, obs_dim: int = 34, hidden: int = 256):
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
            return legal_mask.astype(np.float64) / legal_count
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
            return legal_mask.astype(np.float64) / legal_count
        probs = np.zeros_like(logits)
        probs[0] = 1.0
        return probs
