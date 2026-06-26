from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidCell(nn.Module):
    """A compact liquid time-constant cell.

    Update rule:
        tau = softplus(W_tau[x, h]) + eps
        proposal = tanh(W_in x + W_h h)
        dh = (-h + proposal) / tau
        h_new = h + dt * dh

    This is a lightweight discrete approximation for exploratory regression.
    """

    def __init__(self, input_size: int, hidden_size: int, dt: float = 1.0,
                 tau_min: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.tau_min = tau_min
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.h_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau_proj = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_proj = nn.Linear(input_size + hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.orthogonal_(self.h_proj.weight)
        nn.init.xavier_uniform_(self.tau_proj.weight)
        nn.init.constant_(self.tau_proj.bias, 0.5)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x_t, h], dim=-1)
        tau = F.softplus(self.tau_proj(combined)) + self.tau_min
        proposal = torch.tanh(self.in_proj(x_t) + self.h_proj(h))
        dh = (-h + proposal) / tau
        liquid_state = h + self.dt * dh
        gate = torch.sigmoid(self.gate_proj(combined))
        h_new = gate * liquid_state + (1.0 - gate) * h
        return self.dropout(self.norm(h_new))


class ImprovedLNNRegressor(nn.Module):
    """Improved LNN regressor for tabular questionnaire data.

    Expected input: [batch, sequence_length, input_size].
    For feature-step modelling, input_size=1 and sequence_length=number of features.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 32,
                 output_size: int = 1, dt: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiquidCell(input_size, hidden_size, dt=dt, dropout=dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        batch_size = x_seq.size(0)
        h = torch.zeros(batch_size, self.hidden_size, device=x_seq.device)
        for t in range(x_seq.size(1)):
            h = self.cell(x_seq[:, t, :], h)
        return self.readout(h)


def tabular_to_feature_sequence(X: torch.Tensor) -> torch.Tensor:
    if X.ndim != 2:
        raise ValueError(f"Expected 2D tensor [batch, features], got {tuple(X.shape)}")
    return X.unsqueeze(-1)
