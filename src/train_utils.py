from __future__ import annotations

from dataclasses import dataclass
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.improved_lnn import ImprovedLNNRegressor, tabular_to_feature_sequence


@dataclass
class LNNTrainResult:
    model: ImprovedLNNRegressor
    train_losses: list[float]
    valid_losses: list[float]
    best_epoch: int


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), batch_size=batch_size, shuffle=shuffle)


def train_lnn(X_train, y_train, X_valid, y_valid, hidden_size=32, dropout=0.1, dt=1.0,
              epochs=300, batch_size=32, learning_rate=1e-3, weight_decay=1e-4,
              patience=40, seed=42, device=None) -> LNNTrainResult:
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImprovedLNNRegressor(input_size=1, hidden_size=hidden_size, output_size=1, dt=dt, dropout=dropout).to(device)
    train_loader = make_loader(X_train, y_train, batch_size, True)
    valid_loader = make_loader(X_valid, y_valid, batch_size, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()
    best_state, best_valid, best_epoch, wait = None, float("inf"), -1, 0
    train_losses, valid_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            pred = model(tabular_to_feature_sequence(X_b))
            loss = criterion(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        if valid_loss < best_valid - 1e-6:
            best_valid, best_epoch, wait = valid_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return LNNTrainResult(model, train_losses, valid_losses, best_epoch)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        losses.append(criterion(model(tabular_to_feature_sequence(X_b)), y_b).item())
    return float(np.mean(losses))


@torch.no_grad()
def predict_lnn(model, X, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    return model(tabular_to_feature_sequence(X_t)).detach().cpu().numpy()
